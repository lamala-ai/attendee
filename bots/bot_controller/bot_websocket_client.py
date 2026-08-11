import json
import logging
import time
from queue import Empty, SimpleQueue
from threading import Lock, Thread
from typing import Callable

from websockets import ConnectionClosed
from websockets.sync.client import connect

from bots.container_capacity import log_capacity

logger = logging.getLogger(__name__)


class BotWebsocketClient:
    """
    A websocket loop that sends and receives messages to/from a websocket server
    Designed to be used by a BotController to control the bot audio/video
    remotely.
    """

    # Connection-state values
    NOT_STARTED = "NOT_STARTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"

    def __init__(self, url: str, on_message_callback: Callable[[dict], None]):
        self.on_message_callback = on_message_callback
        self.websocket_url = url
        self.websocket = None

        self.connection_state = self.NOT_STARTED
        self.connection_thread = None
        self.recv_loop_thread = None
        self.send_loop_thread = None
        self.send_queue = SimpleQueue()

        self._max_retries = 30
        self._retry_delay_s = 10
        self.dropped_message_ticker = 0
        self._start_connection_lock = Lock()

        # When the container has no thread left to give, the retry is paced rather than
        # taken on the next frame: this client is started lazily from the audio and video
        # callbacks, so "try again immediately" means thirty attempts a second, each one
        # asking the kernel for the thing it just refused.
        self._thread_start_retry_delay_s = 5
        self._no_thread_start_before = 0.0

    # --------------------------------------------------------------------- #
    #  Public helpers                                                       #
    # --------------------------------------------------------------------- #

    def started(self):
        return self.connection_state != self.NOT_STARTED

    def start(self):
        logger.info(f"Starting BotWebsocketClient for url {self.websocket_url}")
        self._start_connection_thread()

    def cleanup(self):
        logger.info("Stopping BotWebsocketClient")
        try:
            if self.websocket:
                self.websocket.close()
        except Exception as e:
            logger.error("Error closing BotWebsocketClient websocket: %s", e)
        finally:
            self.connection_state = self.STOPPED

    def send_async(self, message: dict):
        if self.connection_state == self.CONNECTED:
            self.send_queue.put(message)
        else:
            if self.dropped_message_ticker % 1000 == 0:
                logger.warning("BotWebsocketClient is not connected, it is in state %s, dropping message", self.connection_state)
            self.dropped_message_ticker += 1

    # --------------------------------------------------------------------- #
    #  Internal helpers                                                     #
    # --------------------------------------------------------------------- #
    def _start_connection_thread(self):
        with self._start_connection_lock:
            if self.connection_state == self.CONNECTING:
                logger.info("BotWebsocketClient connection thread already running")
                return
            if self.connection_thread and self.connection_thread.is_alive():
                logger.info("BotWebsocketClient connection thread already running")
                return
            if time.monotonic() < self._no_thread_start_before:
                return

            self.connection_state = self.CONNECTING
            thread = Thread(target=self._connection_loop, daemon=True)
            try:
                thread.start()
            except RuntimeError as e:
                # The container is out of threads, and this is the failure that used to
                # cost the whole meeting. CONNECTING was already set above, and every
                # route back in refuses to act on a client in that state - the guard at
                # the top of this method, `started()` (which is how the send path decides
                # whether to start us at all), and `_trigger_reconnect`. So the client sat
                # in CONNECTING for ever with no thread behind it, `send_async` dropped
                # every frame from then on, and the bot stayed in the room hearing nothing
                # and saying nothing until somebody restarted the worker.
                #
                # Rolling back to NOT_STARTED is what makes that recoverable: the next
                # frame finds a client that has not started, and tries again. NOT_STARTED
                # rather than whatever it was before, because there is genuinely nothing
                # running now - reporting CONNECTED would queue frames into a socket with
                # no sender behind it.
                self._roll_back_to_not_started()
                self.connection_thread = None
                self._no_thread_start_before = time.monotonic() + self._thread_start_retry_delay_s
                logger.error("BotWebsocketClient could not start its connection thread (%s); retrying in %ss", e, self._thread_start_retry_delay_s)
                log_capacity("BotWebsocketClient could not start a thread")
                return

            self.connection_thread = thread

    def _connection_loop(self):
        retries = 0
        while self.connection_state == self.CONNECTING and retries < self._max_retries:
            try:
                self.websocket = connect(self.websocket_url)

                logger.info("BotWebsocketClient websocket connected, waiting for worker threads to finish")

                # if the worker threads are running, wait for them to finish
                if self.recv_loop_thread and self.recv_loop_thread.is_alive():
                    self.recv_loop_thread.join()
                if self.send_loop_thread and self.send_loop_thread.is_alive():
                    self.send_loop_thread.join()

                self.connection_state = self.CONNECTED
                logger.info("BotWebsocketClient websocket connected, launching worker threads")

                # Launch worker threads (fresh each time we reconnect)
                recv_loop_thread = Thread(target=self.recv_loop, daemon=True)
                send_loop_thread = Thread(target=self.send_loop, daemon=True)
                try:
                    recv_loop_thread.start()
                    self.recv_loop_thread = recv_loop_thread
                    send_loop_thread.start()
                    self.send_loop_thread = send_loop_thread
                except RuntimeError as e:
                    # The same ceiling as in _start_connection_thread, and left alone the
                    # worse half of it: the state above is already CONNECTED, so send_async
                    # would accept every frame of the meeting into a queue that now has no
                    # send loop behind it - silent, and growing. Caught here rather than by
                    # the retry handler below, which would read the state it just set,
                    # decide the loop is finished and return as though this had worked.
                    self._release_connection(e)
                    return
                return  # success – leave the loop
            except Exception as e:
                retries += 1
                logger.warning(
                    "BotWebsocketClient connection attempt %d/%d failed: %s",
                    retries,
                    self._max_retries,
                    e,
                )
                time.sleep(self._retry_delay_s)

        # Handle case where we were stopped before we could connect
        if self.connection_state != self.CONNECTING:
            logger.info("BotWebsocketClient connection loop exited because connection state is %s", self.connection_state)
            return

        # Exhausted retries
        self.connection_state = self.FAILED
        logger.error("BotWebsocketClient failed to establish websocket connection after %d retries", self._max_retries)

    def _roll_back_to_not_started(self):
        """Make the client startable again after a thread it needed could not start.

        Unless it has been stopped. `cleanup()` does not hold `_start_connection_lock`, so
        a bot leaving the meeting can set STOPPED while a start is in flight - and STOPPED
        is the one state that must survive, or a late video frame would raise a websocket
        client back up for a bot that is no longer in the room.
        """
        if self.connection_state == self.STOPPED:
            return
        self.connection_state = self.NOT_STARTED

    def _release_connection(self, error):
        """Let go of a connection we cannot run, leaving the client able to try again.

        NOT_STARTED is the honest state: nothing is connected and nothing is looping, and
        it is the one state the send path will start a client out of. The socket is closed
        rather than dropped because its reader thread is exactly the resource we have just
        run out of, and abandoning it would hold that thread until the far end gave up.
        """
        self._roll_back_to_not_started()
        self._no_thread_start_before = time.monotonic() + self._thread_start_retry_delay_s
        try:
            if self.websocket:
                self.websocket.close()
        except Exception as e:
            logger.warning("BotWebsocketClient error closing the websocket it could not run: %s", e)
        finally:
            self.websocket = None
        logger.error("BotWebsocketClient could not start its worker threads (%s); retrying in %ss", error, self._thread_start_retry_delay_s)
        log_capacity("BotWebsocketClient could not start its worker threads")

    def _trigger_reconnect(self):
        if self.connection_state in [self.CONNECTING, self.FAILED, self.STOPPED]:
            logger.info("BotWebsocketClient aborting websocket reconnect because connection state is %s", self.connection_state)
            return  # already trying or permanently failed
        logger.info("BotWebsocketClient triggering websocket reconnect")
        self._start_connection_thread()

    # --------------------------------------------------------------------- #
    #  Worker threads                                                       #
    # --------------------------------------------------------------------- #
    def send_loop(self):
        logger.info("BotWebsocketClient send loop started")
        while self.connection_state == self.CONNECTED:
            try:
                message = self.send_queue.get(timeout=1)
            except Empty:
                continue  # nothing queued yet

            try:
                self.websocket.send(json.dumps(message))
            except Exception as e:
                logger.info("BotWebsocketClient send failed (%s). Leaving loop.", e)
                break

        logger.info("BotWebsocketClient send loop exited")
        self._trigger_reconnect()

    def recv_loop(self):
        logger.info("BotWebsocketClient recv loop started")
        while self.connection_state == self.CONNECTED:
            try:
                message = self.websocket.recv()
                self.on_message_callback(message)
            except ConnectionClosed:
                logger.info("BotWebsocketClient connection closed. Leaving loop.")
                break
            except Exception as e:
                logger.info("BotWebsocketClient recv failed (%s). Leaving loop.", e)
                break

        logger.info("BotWebsocketClient recv loop exited")
        self._trigger_reconnect()
