"""The worker's size is a decision written down here, not whatever the host happens to report.

Off Kubernetes every bot runs inside a celery pool process, so the pool is both the
deployment's meeting capacity and most of its bill. Left to Celery's defaults it was neither
chosen nor bounded: one process per reported CPU (48 on Railway, 9.2 GB held while idle),
and every meeting past its first hour redelivered as a second copy of its bot. Each test
below fails against those defaults.
"""

import json

from django.test import SimpleTestCase

from attendee.celery import app, broker_transport_options_from

ONE_HOUR = 60 * 60
# The longest bot run in the November 2026 production logs: 17,578 s, a meeting that was
# redelivered every hour on the way there.
LONGEST_OBSERVED_BOT_RUN = 17578


class WorkerPoolSizeTest(SimpleTestCase):
    def test_the_pool_size_is_set_rather_than_taken_from_the_cpu_count(self):
        # None is Celery's "ask os.cpu_count()", which is how a 32-vCPU container got 48.
        self.assertIsNotNone(app.conf.worker_concurrency)
        self.assertLessEqual(app.conf.worker_concurrency, 8)


class BrokerVisibilityTimeoutTest(SimpleTestCase):
    def test_a_meeting_past_its_first_hour_is_not_handed_out_again(self):
        timeout = app.conf.broker_transport_options["visibility_timeout"]
        self.assertGreater(timeout, ONE_HOUR)
        self.assertGreater(timeout, LONGEST_OBSERVED_BOT_RUN)

    def test_the_default_is_a_day(self):
        self.assertEqual(broker_transport_options_from({}), {"visibility_timeout": 24 * ONE_HOUR})

    def test_the_timeout_can_be_set_on_its_own(self):
        options = broker_transport_options_from({"CELERY_BROKER_VISIBILITY_TIMEOUT_SECONDS": "43200"})
        self.assertEqual(options["visibility_timeout"], 43200)

    def test_redis_cluster_options_keep_the_timeout_instead_of_replacing_it(self):
        # CELERY_BROKER_TRANSPORT_OPTIONS used to be the whole dict. Setting it for Redis
        # Cluster must not quietly put the one-hour default back.
        cluster = {"global_keyprefix": "{celeryattendee}:", "fanout_prefix": True, "fanout_patterns": True}
        options = broker_transport_options_from({"CELERY_BROKER_TRANSPORT_OPTIONS": json.dumps(cluster)})
        self.assertEqual(options, {"visibility_timeout": 24 * ONE_HOUR, **cluster})

    def test_the_transport_options_variable_may_still_set_the_timeout_itself(self):
        options = broker_transport_options_from({"CELERY_BROKER_TRANSPORT_OPTIONS": json.dumps({"visibility_timeout": 7200})})
        self.assertEqual(options["visibility_timeout"], 7200)
