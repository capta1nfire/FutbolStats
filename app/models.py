"""Database models using SQLModel."""

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, UniqueConstraint
from sqlmodel import Field, Relationship, SQLModel


class Team(SQLModel, table=True):
    """Team model for both national teams and clubs."""

    __tablename__ = "teams"

    id: Optional[int] = Field(default=None, primary_key=True)
    external_id: int = Field(unique=True, index=True, description="API-Football ID")
    name: str = Field(max_length=255, description="Team name")
    country: Optional[str] = Field(
        default=None, max_length=100, description="Country (for clubs) or NULL (for nationals)"
    )
    team_type: str = Field(
        max_length=50, description="'national' or 'club'"
    )
    logo_url: Optional[str] = Field(default=None, max_length=500, description="Team crest URL")

    # Relationships
    home_matches: list["Match"] = Relationship(
        back_populates="home_team",
        sa_relationship_kwargs={"foreign_keys": "Match.home_team_id"},
    )
    away_matches: list["Match"] = Relationship(
        back_populates="away_team",
        sa_relationship_kwargs={"foreign_keys": "Match.away_team_id"},
    )


class Match(SQLModel, table=True):
    """Match model for storing fixture data."""

    __tablename__ = "matches"

    id: Optional[int] = Field(default=None, primary_key=True)
    external_id: int = Field(unique=True, index=True, description="API-Football fixture ID")
    date: datetime = Field(index=True, description="Match date and time")
    league_id: int = Field(index=True, description="Competition ID")
    season: int = Field(description="Season year")

    home_team_id: int = Field(foreign_key="teams.id", index=True)
    away_team_id: int = Field(foreign_key="teams.id", index=True)

    home_goals: Optional[int] = Field(default=None, description="NULL if not played")
    away_goals: Optional[int] = Field(default=None, description="NULL if not played")

    stats: Optional[dict] = Field(
        default=None, sa_column=Column(JSON), description="Shots, corners, etc."
    )

    status: str = Field(
        max_length=20, default="NS", description="NS, FT, LIVE, etc."
    )
    match_type: str = Field(
        max_length=20, default="official", description="'official' or 'friendly'"
    )
    match_weight: float = Field(
        default=1.0, description="1.0 (official) or 0.6 (friendly)"
    )

    odds_home: Optional[float] = Field(default=None, description="Bookmaker odds for home win")
    odds_draw: Optional[float] = Field(default=None, description="Bookmaker odds for draw")
    odds_away: Optional[float] = Field(default=None, description="Bookmaker odds for away win")

    # Relationships
    home_team: Optional[Team] = Relationship(
        back_populates="home_matches",
        sa_relationship_kwargs={"foreign_keys": "[Match.home_team_id]"},
    )
    away_team: Optional[Team] = Relationship(
        back_populates="away_matches",
        sa_relationship_kwargs={"foreign_keys": "[Match.away_team_id]"},
    )
    predictions: list["Prediction"] = Relationship(back_populates="match")


class Prediction(SQLModel, table=True):
    """Prediction model for storing model predictions."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("match_id", "model_version", name="uq_match_model"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    match_id: int = Field(foreign_key="matches.id", index=True)
    model_version: str = Field(max_length=50, description="e.g., 'v1.0.0'")

    home_prob: float = Field(description="Probability of home win")
    draw_prob: float = Field(description="Probability of draw")
    away_prob: float = Field(description="Probability of away win")

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    match: Optional[Match] = Relationship(back_populates="predictions")

    @property
    def fair_odds_home(self) -> float:
        """Calculate fair odds for home win."""
        return 1 / self.home_prob if self.home_prob > 0 else float("inf")

    @property
    def fair_odds_draw(self) -> float:
        """Calculate fair odds for draw."""
        return 1 / self.draw_prob if self.draw_prob > 0 else float("inf")

    @property
    def fair_odds_away(self) -> float:
        """Calculate fair odds for away win."""
        return 1 / self.away_prob if self.away_prob > 0 else float("inf")
