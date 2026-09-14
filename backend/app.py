"""Legacy entry point — kept so old imports keep working.

The full NetraPath app lives in run_mvp.py (single source of truth).
This module used to define a second, stale Flask app with missing routes
(/api/city/*, /traffic_feed, /video_feed, webcam, uploads) and a divergent
alert schema — that duplication is now removed: we re-export the real app.
"""
from run_mvp import app  # noqa: F401  (single canonical Flask app)

__all__ = ["app"]
