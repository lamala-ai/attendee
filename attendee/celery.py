import json
import os
import ssl

from celery import Celery

# Set the default Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "attendee.settings")

sslCertRequirements = None
if os.getenv("DISABLE_REDIS_SSL"):
    sslCertRequirements = ssl.CERT_NONE
elif os.getenv("REDIS_SSL_REQUIREMENTS"):
    if os.getenv("REDIS_SSL_REQUIREMENTS") == "none":
        sslCertRequirements = ssl.CERT_NONE
    elif os.getenv("REDIS_SSL_REQUIREMENTS") == "optional":
        sslCertRequirements = ssl.CERT_OPTIONAL
    elif os.getenv("REDIS_SSL_REQUIREMENTS") == "required":
        sslCertRequirements = ssl.CERT_REQUIRED

# Create the Celery app
if sslCertRequirements is not None:
    app = Celery(
        "attendee",
        broker_use_ssl={"ssl_cert_reqs": sslCertRequirements},
        redis_backend_use_ssl={"ssl_cert_reqs": sslCertRequirements},
    )
else:
    app = Celery("attendee")

# How long a delivered task may stay unacknowledged before Redis decides its worker died and
# delivers it again. Production acknowledges late (a task is acked when it finishes, so one
# that was running when its worker went away is not lost), and off Kubernetes a bot *is* a
# task that runs for the whole meeting - so with kombu's default of one hour, every meeting
# that passed its first hour was handed out a second time, and again every hour after, each
# copy starting the whole bot over in another pool process. Bots 202 and 203 were each
# started eleven times on 29 Oct 2026.
#
# The window has to outlast the longest meeting, and a bot has no time limit unless it is
# given max_uptime_seconds. It only decides how long a *crashed* worker's tasks wait to be
# redelivered: a clean shutdown (a redeploy) hands unfinished tasks back at once.
DEFAULT_BROKER_VISIBILITY_TIMEOUT_SECONDS = 24 * 60 * 60


def broker_transport_options_from(environ):
    options = {"visibility_timeout": int(environ.get("CELERY_BROKER_VISIBILITY_TIMEOUT_SECONDS", DEFAULT_BROKER_VISIBILITY_TIMEOUT_SECONDS))}

    # Currently the only use case for CELERY_BROKER_TRANSPORT_OPTIONS is to enable support for Redis Cluster hash
    # tags. This is mainly to prevent CROSSSLOT errors when using Redis Cluster (https://github.com/celery/celery/issues/8276#issuecomment-3714489309)
    # For this case set CELERY_BROKER_TRANSPORT_OPTIONS='{"global_keyprefix":"{celeryattendee}:","fanout_prefix":true,"fanout_patterns":true}'
    # Merged over the visibility timeout rather than replacing it, and free to set its own.
    if environ.get("CELERY_BROKER_TRANSPORT_OPTIONS"):
        options.update(json.loads(environ["CELERY_BROKER_TRANSPORT_OPTIONS"]))
    return options


app.conf.update(broker_transport_options=broker_transport_options_from(os.environ))

# Load configuration from Django settings
app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover tasks from all registered Django apps
app.autodiscover_tasks()
