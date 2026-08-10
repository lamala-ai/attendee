"""One managed-policy file, many bots, and what happens when they want different things.

Chrome reads managed policy from a machine-wide directory; off Kubernetes every bot in a
deployment is a celery task in one container. These tests drive the file through several
"processes" by patching the PID the module registers itself under - which is exactly the
axis the old code had no notion of, since it wrote its own policy and returned.
"""

import json
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

from django.test import SimpleTestCase

from bots.web_bot_adapter import chrome_policies

TEAMS_POLICY = {
    "BrowserSwitcherEnabled": True,
    "AlternativeBrowserPath": "/nonexistent-browser",
    "BrowserSwitcherUrlList": ["*", "!microsoft.com"],
}


class ChromePolicyFileTestCase(SimpleTestCase):
    """Points the module at a scratch directory and pretends we are in the image."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.live_pids = set()
        patches = [
            patch.object(chrome_policies, "POLICY_FILE", self.tmp_dir / "attendee-chrome-policies.json"),
            patch.object(chrome_policies, "REGISTRY_FILE", self.tmp_dir / "attendee-chrome-policies.registry.json"),
            patch.object(chrome_policies, "LOCK_FILE", self.tmp_dir / "attendee-chrome-policies.lock"),
            patch.object(chrome_policies, "_is_running_in_the_container", return_value=True),
            patch.object(chrome_policies, "_process_is_alive", side_effect=lambda pid: pid in self.live_pids),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(shutil.rmtree, self.tmp_dir, ignore_errors=True)

    def written_policy(self):
        return json.loads((self.tmp_dir / "attendee-chrome-policies.json").read_text())

    def claim_as(self, pid, policy):
        self.live_pids.add(pid)
        with patch.object(chrome_policies.os, "getpid", return_value=pid):
            return chrome_policies.claim(policy)

    def withdraw_as(self, pid):
        with patch.object(chrome_policies.os, "getpid", return_value=pid):
            return chrome_policies.withdraw()


class TestClaim(ChromePolicyFileTestCase):
    def test_a_single_bot_gets_exactly_what_it_asked_for(self):
        self.claim_as(100, TEAMS_POLICY)

        self.assertEqual(self.written_policy(), TEAMS_POLICY)

    def test_bots_that_agree_get_the_policy_they_agree_on(self):
        self.claim_as(100, TEAMS_POLICY)
        self.claim_as(101, TEAMS_POLICY)

        self.assertEqual(self.written_policy(), TEAMS_POLICY)

    def test_a_disagreement_resolves_to_the_permissive_policy(self):
        """The regression, and the reason this module exists. Against the old behaviour a
        Teams bot starting mid-call wrote its browser-switcher rule over the file every
        other bot's Chrome reads, sending a Google Meet bot away from meet.google.com to
        a browser path that does not exist."""
        self.claim_as(100, {})
        self.claim_as(101, TEAMS_POLICY)

        self.assertEqual(self.written_policy(), {})

    def test_a_meet_bot_joining_does_not_clear_a_lone_teams_bots_policy_by_itself(self):
        """The other direction of the same bug: the old code's last writer won, so a bot
        wanting {} silently disarmed the allowlist for a Teams bot in a live meeting.
        The permissive result here is a *stated* resolution of a real conflict, not a
        write that ignored the other bot - once the Meet bot leaves it is restored."""
        self.claim_as(100, TEAMS_POLICY)
        self.claim_as(101, {})
        self.assertEqual(self.written_policy(), {})

        self.live_pids.discard(101)
        self.withdraw_as(101)

        self.assertEqual(self.written_policy(), TEAMS_POLICY)

    def test_a_bot_that_died_without_withdrawing_stops_counting(self):
        """Nothing in this container reliably runs teardown - that is the premise of the
        whole PR - so a claim is only as good as the process that made it."""
        self.claim_as(100, TEAMS_POLICY)
        self.live_pids.discard(100)

        self.claim_as(101, {})

        self.assertEqual(self.written_policy(), {})

    def test_does_nothing_outside_the_docker_image(self):
        with patch.object(chrome_policies, "_is_running_in_the_container", return_value=False):
            self.assertIsNone(self.claim_as(100, TEAMS_POLICY))

        self.assertFalse((self.tmp_dir / "attendee-chrome-policies.json").exists())

    def test_an_unreadable_registry_does_not_stop_a_bot_claiming(self):
        (self.tmp_dir / "attendee-chrome-policies.registry.json").write_text("{not json")

        self.claim_as(100, TEAMS_POLICY)

        self.assertEqual(self.written_policy(), TEAMS_POLICY)


class TestWithdraw(ChromePolicyFileTestCase):
    def test_the_last_bot_out_leaves_the_permissive_policy(self):
        self.claim_as(100, TEAMS_POLICY)
        self.live_pids.discard(100)
        self.withdraw_as(100)

        self.assertEqual(self.written_policy(), {})

    def test_withdrawing_twice_is_harmless(self):
        self.claim_as(100, TEAMS_POLICY)
        self.withdraw_as(100)
        self.withdraw_as(100)

        self.assertEqual(self.written_policy(), {})
