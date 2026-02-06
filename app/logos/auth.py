"""Authentication for Logos Dashboard endpoints.

DEPRECATED: Use app.security.verify_dashboard_token directly.
Kept as re-export for backward compatibility.
"""

from app.security import verify_dashboard_token  # noqa: F401 — re-export
