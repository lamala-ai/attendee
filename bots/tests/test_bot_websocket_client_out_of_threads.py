"""What the client does when the container has no thread left to give it.

These are regression tests for the failure that made a session look stuck rather than
broken. On 2026-08-11 a bot mid-meeting hit `RuntimeError: can't start new thread` in
`_start_connection_thread`, and because CONNECTING had already been set one line earlier
the client could never be started again by anything: the guard at the top of that method,
`started()` (which is how the send path decides whether to start a client) and
`_trigger_reconnect` all decline to act on a client in that state. `send_async` then
dropped every frame for the rest of the call. The bot stayed in the room, present and
silent, until somebody restarted the worker - which is what a deployment ended up doing,
once per meeting.

Each test below fails against that behaviour.
"""

import unittest
from unittest.mock import Mock, patch

from bots.bot_controller.bot_websocket_client import BotWebsocketClient


class TestBotWebsocketClientOutOfThreads(unittest.TestCase):
    def setUp(self):
        self.client = BotWebsocketClient("ws://test.example.com", Mock())
        # The pacing between retries is real behaviour but not what these tests are about,
        # so it is shrunk to nothing rather than waited out.
        self.client._thread_start_retry_delay_s = 0

    def test_a_client_that_could_not_start_a_thread_can_be_started_again(self):
        """The wedge itself: a failed thread start must leave the client startable.

        Against the old code this fails - connection_state stays CONNECTING for ever, so
        started() answers True and the send path never calls start() again.
        """
        with patch("bots.bot_controller.bot_websocket_client.Thread") as thread_class:
            thread_class.return_value.start.side_effect = RuntimeError("can't start new thread")
            self.client.start()

        self.assertEqual(self.client.connection_state, BotWebsocketClient.NOT_STARTED)
        self.assertFalse(self.client.started(), "a client with no thread behind it must not claim to have started")
        self.assertIsNone(self.client.connection_thread)

    def test_the_next_attempt_actually_starts_a_thread(self):
        """Having failed once, the client connects when the container has room again."""
        with patch("bots.bot_controller.bot_websocket_client.Thread") as thread_class:
            thread_class.return_value.start.side_effect = RuntimeError("can't start new thread")
            self.client.start()

        started = []
        with patch("bots.bot_controller.bot_websocket_client.Thread") as thread_class:
            thread_class.return_value.start.side_effect = lambda: started.append(True)
            thread_class.return_value.is_alive.return_value = True
            self.client.start()

        self.assertEqual(len(started), 1, "the retry never reached Thread.start()")
        self.assertEqual(self.client.connection_state, BotWebsocketClient.CONNECTING)

    def test_a_failed_thread_start_does_not_reach_the_caller(self):
        """The frame callbacks must survive it.

        The RuntimeError used to propagate out through send_per_participant_video and
        into the websockets server's connection handler, which logged "connection handler
        failed" and tore that handler down - so one exhausted moment also cost the inbound
        stream it was raised on.
        """
        with patch("bots.bot_controller.bot_websocket_client.Thread") as thread_class:
            thread_class.return_value.start.side_effect = RuntimeError("can't start new thread")
            self.client.start()  # must not raise

    def test_retries_are_paced_rather_than_taken_on_every_frame(self):
        """A client started from a 30fps video callback must not ask 30 times a second."""
        self.client._thread_start_retry_delay_s = 300

        with patch("bots.bot_controller.bot_websocket_client.Thread") as thread_class:
            thread_class.return_value.start.side_effect = RuntimeError("can't start new thread")
            for _ in range(10):
                self.client.start()
            self.assertEqual(thread_class.return_value.start.call_count, 1, "the failed thread start was retried without waiting")

    def test_a_stopped_client_is_not_resurrected_by_the_rollback(self):
        """STOPPED outranks the rollback.

        cleanup() does not hold the start lock, so a bot leaving the meeting can stop the
        client while a start is in flight. Rolling that back to NOT_STARTED would let the
        next video frame raise a websocket client for a bot no longer in the room.
        """

        def stop_then_fail():
            self.client.connection_state = BotWebsocketClient.STOPPED
            raise RuntimeError("can't start new thread")

        with patch("bots.bot_controller.bot_websocket_client.Thread") as thread_class:
            thread_class.return_value.start.side_effect = stop_then_fail
            self.client.start()

        self.assertEqual(self.client.connection_state, BotWebsocketClient.STOPPED)
        self.assertTrue(self.client.started(), "a stopped client must not look startable again")

    def test_worker_threads_that_cannot_start_do_not_leave_the_client_claiming_to_be_connected(self):
        """The other half of the ceiling, and the worse one.

        _connection_loop sets CONNECTED before launching the recv/send loops. If those
        cannot start, the old code let the RuntimeError fall into the retry handler, which
        read the state it had just set, decided the loop was finished and returned - state
        CONNECTED, no send loop, and send_async accepting every frame of the meeting into
        a queue nothing drains.
        """
        fake_socket = Mock()

        with patch("bots.bot_controller.bot_websocket_client.connect", return_value=fake_socket):
            with patch("bots.bot_controller.bot_websocket_client.Thread") as thread_class:
                thread_class.return_value.start.side_effect = RuntimeError("can't start new thread")
                thread_class.return_value.is_alive.return_value = False
                self.client.connection_state = BotWebsocketClient.CONNECTING
                self.client._connection_loop()

        self.assertEqual(self.client.connection_state, BotWebsocketClient.NOT_STARTED)
        self.assertIsNone(self.client.websocket)
        fake_socket.close.assert_called_once()

        # And nothing is queued into it while it is in that state.
        self.client.send_async({"hello": "world"})
        self.assertTrue(self.client.send_queue.empty())


if __name__ == "__main__":
    unittest.main()
