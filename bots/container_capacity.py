"""What this container is actually running out of, said in numbers.

Off Kubernetes every bot is a celery task in one container, and each one brings two
Xvfb servers, two Chromes, a chromedriver and two ffmpeg encoders with it. What that
exhausts first is neither CPU nor memory - a deployment measured on 2026-08-11 was
sitting at 10 GB of a 32 GB limit and 5.8 of 32 cores when a bot mid-meeting hit:

    RuntimeError: can't start new thread

That is the process/thread ceiling, which is a cgroup limit (`pids.max`) counting
every task in the container - and threads are tasks, so one Chrome counts for dozens.
Nothing in Railway's dashboard draws it, which is why two rounds of debugging went
looking at the two graphs that *are* drawn and found them innocent.

So this module reads the ceiling and how close to it we are, and the two places that
care say so out loud: once as a bot starts, and again on the failure itself, so the
log line that reports a thread that could not start also reports what was left.

Everything here answers None rather than raising. A container whose cgroup files are
laid out differently should lose a diagnostic, not a meeting.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# cgroup v2 keeps one flat hierarchy; v1 keeps a controller per directory. Which one a
# host uses is not ours to choose, so both are read and the first that answers wins.
PIDS_CURRENT_PATHS = (Path("/sys/fs/cgroup/pids.current"), Path("/sys/fs/cgroup/pids/pids.current"))
PIDS_MAX_PATHS = (Path("/sys/fs/cgroup/pids.max"), Path("/sys/fs/cgroup/pids/pids.max"))

# Below this much of the ceiling still free, a bot is starting into a container that
# very likely cannot hold it. Said at WARNING so it appears before the failure does.
HEADROOM_WARNING_FRACTION = 0.2


# libx264 given no -threads sizes its frame threads at roughly 1.5x the CPU count it can
# see, and this container advertises 48 of them (celery's pool comes up 48 wide, one prefork
# worker per visible CPU). So each encoder asks for something like seventy threads - and a
# bot runs two of them, for jobs that are a 5fps capture of a mostly static page and an
# `ultrafast` 1080p screen grab. Two bots then spend more of the container's task budget on
# ffmpeg than on the browsers the meeting actually needs. Nothing here is short of CPU
# (5.8 cores of a 32-core allowance at the worst moment measured), so those threads buy
# nothing and cost the resource that is genuinely scarce.
DEFAULT_ENCODER_THREADS = 2


def encoder_thread_args():
    """`-threads N` for an ffmpeg encoder, as argv, or nothing if a deployment opts out.

    FFMPEG_ENCODER_THREADS=0 (or "auto") hands the decision back to ffmpeg, which is the
    old behaviour and the rollback if some deployment turns out to be encode-bound.
    """
    raw = os.getenv("FFMPEG_ENCODER_THREADS", str(DEFAULT_ENCODER_THREADS)).strip().lower()
    if raw in ("", "0", "auto"):
        return []
    if not raw.isdigit():
        logger.warning(f"FFMPEG_ENCODER_THREADS={raw!r} is not a number, using {DEFAULT_ENCODER_THREADS}")
        return ["-threads", str(DEFAULT_ENCODER_THREADS)]
    return ["-threads", raw]


def _read_first(paths):
    for path in paths:
        try:
            return path.read_text().strip()
        except OSError:
            continue
    return None


def pids_current():
    """Tasks - processes *and* threads - currently charged to this container, or None."""
    raw = _read_first(PIDS_CURRENT_PATHS)
    if raw is None or not raw.isdigit():
        return None
    return int(raw)


def pids_max():
    """The container's task ceiling, or None if there is none or it cannot be read.

    cgroup writes the word "max" for no limit, which is a different answer from "could
    not find out" and must not be reported as a number.
    """
    raw = _read_first(PIDS_MAX_PATHS)
    if raw is None or not raw.isdigit():
        return None
    return int(raw)


def threads_in_container():
    """Threads counted off /proc, for a host that exposes no pids cgroup.

    Slower and racier than the cgroup counter - processes come and go while it walks -
    so it is a fallback rather than the primary reading.
    """
    total = 0
    try:
        entries = [entry for entry in Path("/proc").iterdir() if entry.name.isdigit()]
    except OSError:
        return None

    for entry in entries:
        try:
            status = (entry / "status").read_text()
        except OSError:
            continue  # Exited while we were walking; it is not holding anything now.
        for line in status.splitlines():
            if line.startswith("Threads:"):
                count = line.split()[-1]
                if count.isdigit():
                    total += int(count)
                break

    return total


def capacity_summary():
    """One line naming the ceiling and the distance to it, for a log message.

    Deliberately a string rather than a dict: every caller here is writing it into a
    log line, and a shape nobody parses is a shape nobody has to keep stable.
    """
    current = pids_current()
    limit = pids_max()

    if current is None:
        counted = threads_in_container()
        if counted is None:
            return "container task usage unknown (no pids cgroup and /proc could not be walked)"
        return f"{counted} threads counted in /proc (no pids cgroup to compare against)"

    if limit is None:
        return f"{current} tasks (processes + threads) in use, no pids ceiling set"

    free = limit - current
    return f"{current}/{limit} tasks (processes + threads) in use, {free} left"


def log_capacity(context):
    """Report the headroom, loudly when there is little of it.

    Called where a bot is about to claim its share and where something has just failed
    to. Swallows everything: a container that refuses to join a call because it could
    not measure itself has turned a diagnostic into an outage.
    """
    if os.getenv("DISABLE_CAPACITY_LOGGING", "false").lower() == "true":
        return

    try:
        summary = capacity_summary()
        current = pids_current()
        limit = pids_max()
        low = current is not None and limit is not None and limit > 0 and (limit - current) < limit * HEADROOM_WARNING_FRACTION
        if low:
            logger.warning(f"{context}: {summary} - close to this container's task ceiling, browsers and threads will start failing")
        else:
            logger.info(f"{context}: {summary}")
    except Exception as e:
        logger.warning(f"Could not measure container capacity for {context}: {e}")
