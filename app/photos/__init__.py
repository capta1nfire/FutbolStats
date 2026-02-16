"""Player Photos Pipeline.

Scrapes, validates, processes, and stores HQ player photos
from multiple sources with fallback chain.

Storage: Cloudflare R2 (same bucket as logos, prefix players/)
Sources: Sofascore, API-Football, club sites (future)
Processing: PhotoRoom (background removal), Pillow (crop/thumbnails)
"""

from app.photos.config import get_photos_settings, PhotosSettings

__all__ = ["get_photos_settings", "PhotosSettings"]
