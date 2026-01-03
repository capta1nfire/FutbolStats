"""ETL module for data extraction, transformation, and loading."""

from app.etl.api_football import APIFootballProvider
from app.etl.base import DataProvider
from app.etl.competitions import COMPETITIONS, Competition
from app.etl.pipeline import ETLPipeline

__all__ = [
    "DataProvider",
    "APIFootballProvider",
    "Competition",
    "COMPETITIONS",
    "ETLPipeline",
]
