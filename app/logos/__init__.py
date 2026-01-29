"""3D Logo Generation System.

Handles generation and storage of 3D team logos with perspective variants
for "boxing poster" style matchups.

Variants:
- front_3d: Frontal 3D metallic badge
- facing_right: HOME team (looks at opponent on right)
- facing_left: AWAY team (looks at opponent on left)
- main: Competition logos (single variant)

Storage: Cloudflare R2 with CDN
IA: DALL-E 3 / SDXL (configurable)
"""

from app.logos.config import get_logos_settings, LogosSettings

__all__ = ["get_logos_settings", "LogosSettings"]
