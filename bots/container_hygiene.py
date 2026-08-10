"""Clearing up after bots that died without tidying, before the next one needs the room.

Off Kubernetes every bot in a deployment is a celery task inside one container, and since
every bot got its own webpage_streamer that container holds two Xvfb servers, two Chromes
and a chromedriver per bot instead of one browser in total. All of them share one `/tmp`,
one `/tmp/.X11-unix` and one PID namespace, which turns a bot that died mid-teardown from
its own problem into everybody else's: what it leaves behind is what the *next* bot trips
over.

Both leftovers announce themselves in the same misleading way - as a failure to start
something that has nothing wrong with it:

    Xvfb program closed. ... _XSERVTransMakeAllCOTSServerListeners: server already running

is pyvirtualdisplay walking display numbers and finding a lock file for each, left by an
Xvfb that is no longer running. Nothing is using those displays; nothing can use them
either, because the file that says otherwise outlives the process that wrote it. A bot
that cannot get a display cannot get a browser, and a bot that cannot get a browser never
joins the meeting - so a leak whose origin is a screenshare ends up costing a seat in a
room that never asked to share anything.

Reaping is deliberately conservative: only a lock whose recorded PID is gone, only a
process whose parent is already dead. Nothing here decides that a *running* process has
outlived its usefulness - that judgement belongs to whoever started it, and a container
that kills live bots to tidy up has replaced a slow failure with an immediate one.
"""

import logging
import os
import re
import signal
from pathlib import Path

logger = logging.getLogger(__name__)

X11_SOCKET_DIR = Path("/tmp/.X11-unix")
X_LOCK_GLOB = ".X*-lock"
X_LOCK_PATTERN = re.compile(r"^\.X(\d+)-lock$")

# What an orphan of ours looks like in /proc. chromedriver and Chrome are included
# because a streamer killed without its process group leaves them behind, and a Chrome
# holding a profile directory produces the other misleading startup failure: "session
# not created: probably user data directory is already in use".
ORPHAN_COMMAND_MARKERS = ("run_webpage_streamer.py", "Xvfb", "chromedriver", "chrome")


def process_is_alive(pid):
    """Whether a PID currently names a live process.

    Signal 0 is the ordinary way to ask. EPERM counts as alive: a process we are not
    allowed to signal is emphatically still there, and treating it as dead would mean
    deleting the lock of a display that is genuinely in use.
    """
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _pid_in_x_lock(lock_path):
    """The PID an X lock file records, or None if it does not read like one.

    Xvfb writes the server's PID as a right-aligned eleven-character decimal. Anything
    else is not a file we wrote, and is left alone.
    """
    try:
        contents = lock_path.read_text()
    except OSError:
        return None
    stripped = contents.strip()
    if not stripped.isdigit():
        return None
    return int(stripped)


def reap_stale_x_locks(tmp_dir=Path("/tmp"), socket_dir=X11_SOCKET_DIR):
    """Delete the lock and socket of every X display whose server is gone.

    Returns the display numbers reclaimed, so a caller can log what it freed rather than
    reporting that it ran.
    """
    reclaimed = []
    try:
        locks = sorted(tmp_dir.glob(X_LOCK_GLOB))
    except OSError as e:
        logger.warning(f"Could not list X display locks in {tmp_dir}: {e}")
        return reclaimed

    for lock_path in locks:
        match = X_LOCK_PATTERN.match(lock_path.name)
        if not match:
            continue
        pid = _pid_in_x_lock(lock_path)
        if pid is None or process_is_alive(pid):
            continue

        display_number = match.group(1)
        try:
            lock_path.unlink()
        except FileNotFoundError:
            continue
        except OSError as e:
            # Someone else's to delete, or a read-only /tmp. Either way this display
            # stays unusable, and saying which one beats a silent skip.
            logger.warning(f"Could not remove stale X lock {lock_path}: {e}")
            continue

        socket_path = socket_dir / f"X{display_number}"
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass
        except OSError as e:
            logger.warning(f"Removed {lock_path} but could not remove its socket {socket_path}: {e}")

        reclaimed.append(int(display_number))

    if reclaimed:
        logger.info(f"Reclaimed X displays left locked by processes that are gone: {reclaimed}")
    return reclaimed


def _proc_command_line(pid, proc_dir):
    try:
        raw = (proc_dir / str(pid) / "cmdline").read_bytes()
    except OSError:
        return None
    return raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()


def _proc_parent_pid(pid, proc_dir):
    """The PPID from /proc/<pid>/stat, or None if it cannot be read.

    Parsed from the field after the executable name rather than by splitting the whole
    line: a process name can contain spaces and parentheses, and every field before the
    closing parenthesis is therefore unsafe to count.
    """
    try:
        stat_line = (proc_dir / str(pid) / "stat").read_text()
    except OSError:
        return None
    end_of_name = stat_line.rfind(")")
    if end_of_name == -1:
        return None
    fields = stat_line[end_of_name + 2 :].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def find_orphaned_bot_processes(proc_dir=Path("/proc")):
    """PIDs of our browser-side processes whose owner is no longer running.

    "Orphaned" is read off the parent, not off any bookkeeping of ours: a streamer, Xvfb
    or Chrome started by a bot has that bot's celery worker as its parent for as long as
    the worker lives, and is reparented to PID 1 the moment the worker dies. So a parent
    of 1 means nothing in this container is still waiting on it.

    Bots run in prefork children, never in the celery main process, so nothing legitimate
    here is a direct child of PID 1 - which is what makes the test safe to act on.
    """
    orphans = []
    try:
        entries = [entry for entry in proc_dir.iterdir() if entry.name.isdigit()]
    except OSError as e:
        logger.warning(f"Could not scan {proc_dir} for orphaned processes: {e}")
        return orphans

    for entry in entries:
        pid = int(entry.name)
        if pid <= 1 or pid == os.getpid():
            continue
        if _proc_parent_pid(pid, proc_dir) != 1:
            continue
        command = _proc_command_line(pid, proc_dir)
        if not command:
            continue
        if not any(marker in command for marker in ORPHAN_COMMAND_MARKERS):
            continue
        orphans.append(pid)

    return orphans


def reap_orphaned_bot_processes(proc_dir=Path("/proc")):
    """SIGKILL the browser-side processes no bot is waiting on any more.

    Straight to SIGKILL rather than SIGTERM-then-wait: these have no teardown left worth
    running - whatever they were rendering has no bot to render it for - and a polite
    signal to a wedged Chrome is how they came to be sitting here in the first place.

    Signalled individually rather than by process group. The group of an orphan can have
    been inherited by anything, up to and including this container's celery worker, and
    a tidy-up that can SIGKILL the process running every other meeting is worse than the
    mess it is clearing.
    """
    reaped = []
    for pid in find_orphaned_bot_processes(proc_dir=proc_dir):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue
        except PermissionError as e:
            logger.warning(f"Not allowed to reap orphaned process {pid}: {e}")
            continue
        reaped.append(pid)

    if reaped:
        logger.info(f"Reaped browser-side processes left behind by bots that are gone: {reaped}")
    return reaped


def tidy_up_after_departed_bots():
    """Both sweeps, best-effort, for a bot that is about to need a display and a browser.

    Swallows everything. This runs on the path that starts a meeting, and a container
    that refuses to join a call because it could not tidy up has turned a leak it was
    coping with into an outage.
    """
    if os.getenv("DISABLE_CONTAINER_HYGIENE", "false").lower() == "true":
        logger.info("Container hygiene is disabled, not reaping stale displays or orphaned processes")
        return

    try:
        reap_stale_x_locks()
    except Exception as e:
        logger.warning(f"Error reaping stale X display locks: {e}")

    try:
        reap_orphaned_bot_processes()
    except Exception as e:
        logger.warning(f"Error reaping orphaned bot processes: {e}")
