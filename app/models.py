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


class PredictionOutcome(SQLModel, table=True):
    """
    Stores the outcome of each prediction after the match is played.
    Links predictions to actual results for model evaluation.
    """

    __tablename__ = "prediction_outcomes"

    id: Optional[int] = Field(default=None, primary_key=True)
    prediction_id: int = Field(foreign_key="predictions.id", unique=True, index=True)
    match_id: int = Field(foreign_key="matches.id", index=True)

    # Actual result
    actual_result: str = Field(
        max_length=10, description="'home', 'draw', or 'away'"
    )
    actual_home_goals: int = Field(description="Final home goals")
    actual_away_goals: int = Field(description="Final away goals")

    # Predicted result (highest probability)
    predicted_result: str = Field(
        max_length=10, description="'home', 'draw', or 'away'"
    )
    prediction_correct: bool = Field(description="Did we predict correctly?")

    # Confidence metrics
    confidence: float = Field(description="Highest probability value (0-1)")
    confidence_tier: str = Field(
        max_length=10, description="'gold', 'silver', or 'copper'"
    )

    # xG data from API-Football
    xg_home: Optional[float] = Field(default=None, description="Expected goals home")
    xg_away: Optional[float] = Field(default=None, description="Expected goals away")
    xg_diff: Optional[float] = Field(default=None, description="xG home - xG away")

    # Disruption factors
    had_red_card: bool = Field(default=False, description="Was there a red card?")
    had_penalty: bool = Field(default=False, description="Was there a penalty?")
    had_var_decision: bool = Field(default=False, description="VAR overturned decision?")
    red_card_minute: Optional[int] = Field(default=None, description="Minute of first red card")

    # Match context
    home_possession: Optional[float] = Field(default=None, description="Home possession %")
    total_shots_home: Optional[int] = Field(default=None)
    total_shots_away: Optional[int] = Field(default=None)
    shots_on_target_home: Optional[int] = Field(default=None)
    shots_on_target_away: Optional[int] = Field(default=None)

    # Timing
    audited_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    prediction: Optional[Prediction] = Relationship()
    match: Optional[Match] = Relationship()
    audit: Optional["PostMatchAudit"] = Relationship(back_populates="outcome")


class PostMatchAudit(SQLModel, table=True):
    """
    Detailed analysis of prediction failures/successes.
    Classifies deviations and identifies root causes.
    """

    __tablename__ = "post_match_audits"

    id: Optional[int] = Field(default=None, primary_key=True)
    outcome_id: int = Field(foreign_key="prediction_outcomes.id", unique=True, index=True)

    # Deviation classification
    deviation_type: str = Field(
        max_length=20,
        description="'minimal', 'expected', 'anomaly'"
    )
    deviation_score: float = Field(
        description="0-1 score indicating how unexpected the result was"
    )

    # Root cause analysis
    primary_factor: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Main factor explaining deviation (red_card, penalty, xg_mismatch, etc.)"
    )
    secondary_factors: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Additional contributing factors"
    )

    # xG analysis
    xg_result_aligned: bool = Field(
        description="Did xG predict the same winner as actual result?"
    )
    xg_prediction_aligned: bool = Field(
        description="Did our prediction align with xG?"
    )
    goals_vs_xg_home: Optional[float] = Field(
        default=None, description="actual_goals - xG for home"
    )
    goals_vs_xg_away: Optional[float] = Field(
        default=None, description="actual_goals - xG for away"
    )

    # Learning signals
    should_adjust_model: bool = Field(
        default=False,
        description="Flag for model recalibration"
    )
    adjustment_notes: Optional[str] = Field(
        default=None, max_length=500, description="Notes for model improvement"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Relationships
    outcome: Optional[PredictionOutcome] = Relationship(back_populates="audit")


class ModelPerformanceLog(SQLModel, table=True):
    """
    Weekly aggregated metrics for model performance tracking.
    Used for trend analysis and triggering recalibration.
    """

    __tablename__ = "model_performance_logs"
    __table_args__ = (
        UniqueConstraint("model_version", "week_start", name="uq_model_week"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    model_version: str = Field(max_length=50, index=True)
    week_start: datetime = Field(index=True, description="Start of the week (Monday)")

    # Prediction counts
    total_predictions: int = Field(default=0)
    correct_predictions: int = Field(default=0)
    accuracy: float = Field(default=0.0, description="correct / total")

    # By confidence tier
    gold_total: int = Field(default=0)
    gold_correct: int = Field(default=0)
    gold_accuracy: float = Field(default=0.0)

    silver_total: int = Field(default=0)
    silver_correct: int = Field(default=0)
    silver_accuracy: float = Field(default=0.0)

    copper_total: int = Field(default=0)
    copper_correct: int = Field(default=0)
    copper_accuracy: float = Field(default=0.0)

    # Calibration metrics
    brier_score: Optional[float] = Field(default=None, description="Lower is better")
    log_loss: Optional[float] = Field(default=None)

    # Deviation analysis
    anomaly_count: int = Field(default=0, description="Number of anomaly deviations")
    anomaly_rate: float = Field(default=0.0)

    # Value bet performance
    value_bets_placed: int = Field(default=0)
    value_bets_won: int = Field(default=0)
    value_bet_roi: Optional[float] = Field(default=None, description="Return on investment %")

    created_at: datetime = Field(default_factory=datetime.utcnow)


class TeamAdjustment(SQLModel, table=True):
    """
    Per-team confidence adjustments based on prediction anomalies.
    Used to calibrate predictions for teams with consistent deviations.
    """

    __tablename__ = "team_adjustments"

    id: Optional[int] = Field(default=None, primary_key=True)
    team_id: int = Field(foreign_key="teams.id", unique=True, index=True)

    # Confidence adjustment (1.0 = no change, 0.9 = reduce home confidence 10%)
    confidence_multiplier: float = Field(
        default=1.0,
        description="Multiplier for team's win probability (0.5-2.0)"
    )

    # Metrics justifying the adjustment
    total_predictions: int = Field(default=0)
    correct_predictions: int = Field(default=0)
    anomaly_count: int = Field(default=0)
    avg_deviation_score: float = Field(default=0.0)

    # Control
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    adjustment_reason: Optional[str] = Field(
        default=None, max_length=200, description="Why this adjustment was applied"
    )

    # Relationship
    team: Optional[Team] = Relationship()


class ModelSnapshot(SQLModel, table=True):
    """
    Model version snapshots for rollback capability.
    Prevents deploying models worse than the baseline.
    """

    __tablename__ = "model_snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    model_version: str = Field(max_length=50, index=True)

    # Model file path
    model_path: str = Field(max_length=500, description="Path to saved model file")

    # Validation metrics
    brier_score: float = Field(description="Cross-validation Brier score")
    cv_brier_scores: Optional[dict] = Field(
        default=None, sa_column=Column(JSON), description="Per-fold CV scores"
    )
    samples_trained: int = Field(description="Number of training samples")

    # Status
    is_active: bool = Field(default=False, index=True, description="Currently deployed model")
    is_baseline: bool = Field(
        default=False, description="Reference model for comparison (Brier: 0.2063)"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    training_config: Optional[dict] = Field(
        default=None, sa_column=Column(JSON), description="Hyperparameters used"
    )
