"""The one Chrome managed-policy file, written by every bot in the container at once.

Chrome reads managed policy from a machine-wide directory, and the image points one file
in it at `/tmp/attendee-chrome-policies.json` (see the symlink in the Dockerfile). There
is therefore exactly one such file per container, however many bots are in it - and off
Kubernetes there are many, because every bot is a celery task rather than a pod of its
own.

Each bot used to write that file with only itself in mind, at `init_driver()` time. The
last bot to start a browser decided, retroactively, what every already-running bot's
Chrome would read on its next policy refresh. With the default settings nothing is lost:
`subclass_specific_chrome_policies()` returns `{}` for every adapter and each write is
identical. Turn `ENFORCE_DOMAIN_ALLOWLIST_IN_CHROME` on and it stops being harmless - a
Teams bot joining mid-call hands every other bot's Chrome a browser-switcher rule that
sends everything outside microsoft.com to `/nonexistent-browser`, and a Google Meet bot
in the middle of a meeting is looking at a URL that rule redirects away from.

So the file is written from what the *live* bots each want rather than from whoever wrote
last. One entry per process - a celery prefork child runs one bot at a time - pruned by
liveness, so a bot killed without teardown stops counting on its own.

Where they disagree, the permissive policy wins. That is a real choice and worth stating:
the allowlist is defence in depth for a Teams bot, and imposing it on a Meet bot costs
that bot its meeting outright. A lost guard rail beats a lost seat, and the disagreement
is logged rather than silently resolved.

A container running one bot - Kubernetes, or a single-bot deployment - has nothing to
disagree with and gets exactly the behaviour it had before.
"""

import fcntl
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

MANAGED_POLICY_SYMLINK = Path("/etc/opt/chrome/policies/managed/attendee-chrome-policies.json")
POLICY_FILE = Path("/tmp/attendee-chrome-policies.json")
REGISTRY_FILE = Path("/tmp/attendee-chrome-policies.registry.json")
LOCK_FILE = Path("/tmp/attendee-chrome-policies.lock")

PERMISSIVE_POLICY = {}


def _process_is_alive(pid):
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_registry():
    try:
        contents = json.loads(REGISTRY_FILE.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as e:
        # A registry we cannot read is worse than none: it would pin every bot to
        # whatever the unreadable file was thought to say. Start again from empty and
        # let the live bots re-register on their next browser launch.
        logger.warning(f"Could not read the Chrome policy registry, starting a fresh one: {e}")
        return {}
    if not isinstance(contents, dict):
        return {}
    return contents


def _live_entries(registry):
    return {owner: policy for owner, policy in registry.items() if owner.isdigit() and _process_is_alive(int(owner))}


def _merge(registry):
    """The policy to write for a set of live claims.

    Identical claims - the overwhelmingly common case, since every adapter but Teams asks
    for `{}` - are that claim. Anything else is a disagreement no single file can honour.
    """
    policies = list(registry.values())
    if not policies:
        return PERMISSIVE_POLICY
    first = policies[0]
    if all(policy == first for policy in policies[1:]):
        return first
    logger.warning(f"Bots in this container want different Chrome policies ({policies}); writing the permissive one so no bot is redirected away from its own meeting")
    return PERMISSIVE_POLICY


def _rewrite(mutate):
    """Apply `mutate` to the registry and rewrite both files, under an exclusive lock.

    The lock is held across the read, the mutation and both writes because the readers
    are other containers' worth of bots in this one, arriving whenever somebody starts a
    browser. Its own file, not the policy file: taking a lock on the file Chrome reads
    would mean truncating it while a browser might be part-way through reading it.
    """
    try:
        LOCK_FILE.touch(exist_ok=True)
        lock_handle = LOCK_FILE.open("r+")
    except OSError as e:
        logger.warning(f"Could not open the Chrome policy lock, leaving the policy file alone: {e}")
        return None

    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        registry = _live_entries(_read_registry())
        mutate(registry)
        policy = _merge(registry)
        POLICY_FILE.write_text(json.dumps(policy, indent=2))
        REGISTRY_FILE.write_text(json.dumps(registry))
        return policy
    except OSError as e:
        logger.warning(f"Could not write the Chrome policy file: {e}")
        return None
    finally:
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_UN)
        finally:
            lock_handle.close()


def _is_running_in_the_container():
    # No symlink, no managed-policy directory to feed - we are not in the docker image,
    # and writing /tmp files nothing reads would only be confusing.
    return MANAGED_POLICY_SYMLINK.is_symlink()


def claim(policy):
    """Register this process's policy and rewrite the shared file to suit every live bot."""
    if not _is_running_in_the_container():
        logger.warning("Attendee chrome policy file symlink does not exist, skipping writing chrome policies.")
        return None

    def mutate(registry):
        registry[str(os.getpid())] = policy

    written = _rewrite(mutate)
    if written is not None:
        logger.info("Chrome policy file written to %s: %s (this bot asked for %s)", POLICY_FILE, written, policy)
    return written


def withdraw():
    """Drop this process's claim, so a policy only it wanted stops applying to the rest."""
    if not _is_running_in_the_container():
        return None

    def mutate(registry):
        registry.pop(str(os.getpid()), None)

    return _rewrite(mutate)
