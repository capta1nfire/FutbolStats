"""Database models using SQLModel."""

from datetime import date, datetime
from typing import Optional
from uuid import UUID

from sqlalchemy import JSON, Column, DateTime, LargeBinary, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID
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

    # Wikipedia / Wikidata linkage (optional, admin-managed)
    wiki_url: Optional[str] = Field(
        default=None,
        sa_column_kwargs={"nullable": True},
        description="User-provided Wikipedia URL (source input)",
    )
    wikidata_id: Optional[str] = Field(
        default=None,
        max_length=20,
        sa_column_kwargs={"nullable": True},
        description="Wikidata Q-number (e.g., Q12345)",
    )
    wiki_title: Optional[str] = Field(
        default=None,
        max_length=255,
        sa_column_kwargs={"nullable": True},
        description="Canonical Wikipedia title (derived)",
    )
    wiki_lang: Optional[str] = Field(
        default=None,
        max_length=32,
        sa_column_kwargs={"nullable": True},
        description="Wikipedia language/project code (derived, e.g., en, es, pt-br)",
    )
    wiki_url_cached: Optional[str] = Field(
        default=None,
        sa_column_kwargs={"nullable": True},
        description="Canonical Wikipedia URL (derived)",
    )
    wiki_source: Optional[str] = Field(
        default=None,
        max_length=32,
        sa_column_kwargs={"nullable": True},
        description="How the wiki link was established (manual, api, fuzzy, etc.)",
    )
    wiki_confidence: Optional[float] = Field(
        default=None,
        sa_column_kwargs={"nullable": True},
        description="Match confidence in [0,1] (derived)",
    )
    wiki_matched_at: Optional[datetime] = Field(
        default=None,
        sa_column_kwargs={"nullable": True},
        description="When the wiki link was last established/updated (UTC)",
    )

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
    round: Optional[str] = Field(
        default=None,
        max_length=80,
        description="API-Football fixture.league.round (e.g., 'Regular Season - 21')",
    )

    home_team_id: int = Field(foreign_key="teams.id", index=True)
    away_team_id: int = Field(foreign_key="teams.id", index=True)

    home_goals: Optional[int] = Field(default=None, description="NULL if not played")
    away_goals: Optional[int] = Field(default=None, description="NULL if not played")

    stats: Optional[dict] = Field(
        default=None, sa_column=Column(JSON), description="Shots, corners, etc."
    )
    events: Optional[list] = Field(
        default=None, sa_column=Column(JSON), description="Match events (goals, cards, etc.)"
    )

    status: str = Field(
        max_length=20, default="NS", description="NS, FT, LIVE, etc."
    )
    elapsed: Optional[int] = Field(
        default=None, description="Current minute for live matches (from API)"
    )
    elapsed_extra: Optional[int] = Field(
        default=None, description="Added/injury time minutes (e.g., 3 for 90+3)"
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
    odds_recorded_at: Optional[datetime] = Field(default=None, description="When odds were last recorded")

    # Historical/backfill odds (FDUK, OddsPortal, Sofascore)
    opening_odds_home: Optional[float] = Field(default=None)
    opening_odds_draw: Optional[float] = Field(default=None)
    opening_odds_away: Optional[float] = Field(default=None)
    opening_odds_source: Optional[str] = Field(default=None, max_length=100)
    opening_odds_kind: Optional[str] = Field(default=None, max_length=50)
    opening_odds_column: Optional[str] = Field(default=None, max_length=50)
    opening_odds_recorded_at: Optional[datetime] = Field(default=None)
    opening_odds_recorded_at_type: Optional[str] = Field(default=None, max_length=50)

    # Venue information
    venue_name: Optional[str] = Field(default=None, max_length=255, description="Stadium name")
    venue_city: Optional[str] = Field(default=None, max_length=100, description="Stadium city")

    # Fast-path narrative tracking
    finished_at: Optional[datetime] = Field(default=None, description="When match finished (FT/AET/PEN detected)")
    stats_ready_at: Optional[datetime] = Field(default=None, description="When stats passed gating requirements")
    stats_last_checked_at: Optional[datetime] = Field(default=None, description="Last stats refresh attempt")

    # Data quality flags (for training/backtest exclusion)
    tainted: bool = Field(default=False, description="Exclude from training/backtest due to data quality issues")
    tainted_reason: Optional[str] = Field(default=None, max_length=100, description="Reason for taint flag")

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

    # Phase 2: Point-in-Time anchor — defines the temporal boundary of ALL
    # information used to generate this prediction (odds, lineup, ratings).
    # Compare model vs market at the SAME asof_timestamp for CLV auditing.
    asof_timestamp: Optional[datetime] = Field(
        default=None,
        sa_column=Column(DateTime(timezone=True), nullable=True),
        description="PIT anchor: all inputs recorded_at <= this timestamp"
    )

    # Rerun tracking (nullable for legacy predictions)
    run_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), nullable=True, index=True),
        description="UUID of the prediction rerun that generated this prediction"
    )

    # Frozen prediction fields - preserves original prediction before match starts
    is_frozen: bool = Field(default=False, description="Whether prediction is locked")
    frozen_at: Optional[datetime] = Field(default=None, description="When prediction was frozen")

    # Frozen bookmaker odds at freeze time (for EV calculation proof)
    frozen_odds_home: Optional[float] = Field(default=None, description="Home odds at freeze time")
    frozen_odds_draw: Optional[float] = Field(default=None, description="Draw odds at freeze time")
    frozen_odds_away: Optional[float] = Field(default=None, description="Away odds at freeze time")

    # Frozen EV calculations at freeze time
    frozen_ev_home: Optional[float] = Field(default=None, description="Home EV at freeze time")
    frozen_ev_draw: Optional[float] = Field(default=None, description="Draw EV at freeze time")
    frozen_ev_away: Optional[float] = Field(default=None, description="Away EV at freeze time")

    # Frozen confidence tier at freeze time
    frozen_confidence_tier: Optional[str] = Field(
        default=None, max_length=10, description="Confidence tier at freeze time"
    )

    # Frozen value bets JSON (preserves which bets had value at freeze time)
    frozen_value_bets: Optional[dict] = Field(
        default=None, sa_column=Column(JSON), description="Value bets snapshot at freeze time"
    )

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

    # Narrative insights (human-readable explanations)
    narrative_insights: Optional[list] = Field(
        default=None,
        sa_column=Column(JSON),
        description="List of narrative insight objects with type, icon, message, priority"
    )
    momentum_analysis: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Momentum analysis object with type, icon, message"
    )

    # LLM-generated narrative (RunPod/Qwen)
    llm_narrative_json: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Full LLM-generated narrative JSON"
    )
    llm_narrative_status: Optional[str] = Field(
        default=None,
        max_length=20,
        description="ok, error, or disabled"
    )
    llm_narrative_generated_at: Optional[datetime] = Field(
        default=None,
        description="When the LLM narrative was generated"
    )
    llm_narrative_model: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Model used (e.g., qwen-vllm)"
    )
    llm_narrative_delay_ms: Optional[int] = Field(
        default=None,
        description="RunPod queue delay in milliseconds"
    )
    llm_narrative_exec_ms: Optional[int] = Field(
        default=None,
        description="RunPod execution time in milliseconds"
    )
    llm_narrative_tokens_in: Optional[int] = Field(
        default=None,
        description="Input tokens used"
    )
    llm_narrative_tokens_out: Optional[int] = Field(
        default=None,
        description="Output tokens generated"
    )
    llm_narrative_worker_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="RunPod worker ID"
    )
    llm_narrative_error_code: Optional[str] = Field(
        default=None,
        max_length=50,
        description="Error code: runpod_http_error, runpod_timeout, schema_invalid, json_parse_error, gating_skipped, empty_output, unknown"
    )
    llm_narrative_error_detail: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Error detail/exception message (truncated)"
    )
    llm_narrative_request_id: Optional[str] = Field(
        default=None,
        max_length=100,
        description="RunPod job ID for correlation"
    )
    llm_narrative_attempts: Optional[int] = Field(
        default=None,
        description="Number of generation attempts (1 or 2)"
    )

    # LLM Traceability (for debugging hallucinations)
    llm_prompt_version: Optional[str] = Field(
        default=None,
        max_length=20,
        description="Prompt template version (e.g., 'v1.0', 'v1.1')"
    )
    llm_prompt_input_json: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Sanitized payload sent to LLM (no secrets)"
    )
    llm_prompt_input_hash: Optional[str] = Field(
        default=None,
        max_length=64,
        description="SHA256 of canonicalized input JSON"
    )
    llm_output_raw: Optional[str] = Field(
        default=None,
        description="Raw LLM output before parsing"
    )
    llm_validation_errors: Optional[list] = Field(
        default=None,
        sa_column=Column(JSON),
        description="List of claim validation errors detected"
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
    Per-team confidence adjustments with contextual intelligence.
    Supports home/away split, recovery factor, and international context.
    """

    __tablename__ = "team_adjustments"

    id: Optional[int] = Field(default=None, primary_key=True)
    team_id: int = Field(foreign_key="teams.id", unique=True, index=True)

    # Legacy field (kept for compatibility)
    confidence_multiplier: float = Field(
        default=1.0,
        description="Combined multiplier (deprecated, use home/away)"
    )

    # Home/Away split multipliers
    home_multiplier: float = Field(
        default=1.0,
        description="Multiplier when team plays at home (0.5-2.0)"
    )
    away_multiplier: float = Field(
        default=1.0,
        description="Multiplier when team plays away (0.5-2.0)"
    )

    # Recovery tracking (3 consecutive MINIMAL = +0.02 forgiveness)
    consecutive_minimal_count: int = Field(
        default=0,
        description="Consecutive MINIMAL audits for recovery"
    )
    last_anomaly_date: Optional[datetime] = Field(
        default=None,
        description="Date of last anomaly for decay tracking"
    )

    # International context penalty
    international_penalty: float = Field(
        default=1.0,
        description="Penalty for international commitments (0.8-1.0)"
    )

    # Overall metrics
    total_predictions: int = Field(default=0)
    correct_predictions: int = Field(default=0)
    anomaly_count: int = Field(default=0)
    avg_deviation_score: float = Field(default=0.0)

    # Home-specific metrics
    home_predictions: int = Field(default=0)
    home_correct: int = Field(default=0)
    home_anomalies: int = Field(default=0)

    # Away-specific metrics
    away_predictions: int = Field(default=0)
    away_correct: int = Field(default=0)
    away_anomalies: int = Field(default=0)

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

    The model_blob field stores the XGBoost model as binary data (BYTEA),
    enabling fast loading from PostgreSQL without filesystem dependency.
    """

    __tablename__ = "model_snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    model_version: str = Field(max_length=50, index=True)

    # Model binary data (BYTEA) - stores XGBoost model via save_raw()
    model_blob: Optional[bytes] = Field(
        default=None,
        sa_column=Column(LargeBinary),
        description="XGBoost model binary (from save_raw())"
    )

    # Legacy model file path (kept for compatibility, now optional)
    model_path: Optional[str] = Field(
        default=None, max_length=500, description="Path to saved model file (legacy)"
    )

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


class OddsHistory(SQLModel, table=True):
    """
    Historical odds snapshots for tracking line movements.

    Stores odds at different points in time to enable:
    - Pre-match odds analysis (opening vs closing)
    - Line movement detection (steam moves, sharp action)
    - Backtesting value bets with actual closing odds
    - Future: multi-bookmaker comparison
    """

    __tablename__ = "odds_history"

    id: Optional[int] = Field(default=None, primary_key=True)
    match_id: int = Field(foreign_key="matches.id", index=True)
    recorded_at: datetime = Field(default_factory=datetime.utcnow, description="When snapshot was taken")

    # Anti-lookahead timestamps (P0.2 Telemetry)
    observed_at: Optional[datetime] = Field(
        default=None,
        description="When odds were observed at the source (provider timestamp if available)"
    )
    ingested_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When odds were ingested into our system"
    )

    # Odds values
    odds_home: Optional[float] = Field(default=None)
    odds_draw: Optional[float] = Field(default=None)
    odds_away: Optional[float] = Field(default=None)

    # Metadata
    source: str = Field(default="api_football", max_length=50, description="Bookmaker source")
    is_opening: bool = Field(default=False, description="First recorded odds for this match")
    is_closing: bool = Field(default=False, description="Last odds before kickoff")

    # Data quality flags (P0 Telemetry)
    quarantined: bool = Field(default=False, description="Excluded from training/backtest due to validation failure")
    quarantine_reason: Optional[str] = Field(default=None, max_length=100, description="Reason for quarantine")
    tainted: bool = Field(default=False, description="Potentially contaminated by lookahead bias")
    taint_reason: Optional[str] = Field(default=None, max_length=100, description="Reason for taint flag")

    # Computed fields for quick analysis
    implied_home: Optional[float] = Field(default=None, description="1/odds_home")
    implied_draw: Optional[float] = Field(default=None, description="1/odds_draw")
    implied_away: Optional[float] = Field(default=None, description="1/odds_away")
    overround: Optional[float] = Field(default=None, description="Sum of implied probs (margin)")

    @classmethod
    def from_odds(
        cls,
        match_id: int,
        odds_home: Optional[float],
        odds_draw: Optional[float],
        odds_away: Optional[float],
        source: str = "api_football",
        is_opening: bool = False,
        is_closing: bool = False,
    ) -> "OddsHistory":
        """Create an OddsHistory entry with computed fields."""
        implied_home = 1 / odds_home if odds_home and odds_home > 0 else None
        implied_draw = 1 / odds_draw if odds_draw and odds_draw > 0 else None
        implied_away = 1 / odds_away if odds_away and odds_away > 0 else None

        overround = None
        if implied_home and implied_draw and implied_away:
            overround = implied_home + implied_draw + implied_away

        return cls(
            match_id=match_id,
            odds_home=odds_home,
            odds_draw=odds_draw,
            odds_away=odds_away,
            source=source,
            is_opening=is_opening,
            is_closing=is_closing,
            implied_home=implied_home,
            implied_draw=implied_draw,
            implied_away=implied_away,
            overround=overround,
        )


class PITReport(SQLModel, table=True):
    """
    PIT (Point-In-Time) evaluation reports stored in DB for persistence.
    Survives Railway deploys (unlike filesystem logs/).
    """

    __tablename__ = "pit_reports"
    __table_args__ = (
        UniqueConstraint("report_type", "report_date", name="uq_pit_report_type_date"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    report_type: str = Field(
        max_length=20, index=True, description="'daily' or 'weekly'"
    )
    report_date: datetime = Field(
        index=True, description="Date of the report (UTC, date only)"
    )
    payload: dict = Field(
        sa_column=Column(JSON), description="Full JSON report content"
    )
    source: str = Field(
        default="scheduler", max_length=50, description="Origin: scheduler, manual, script"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class AlphaProgressSnapshot(SQLModel, table=True):
    """
    Alpha Progress Snapshots - tracks evolution of "Progreso hacia Re-test/Alpha".
    Allows auditing progress over time without screenshots.
    Survives Railway deploys (DB-backed).
    """

    __tablename__ = "alpha_progress_snapshots"

    id: Optional[int] = Field(default=None, primary_key=True)
    captured_at: datetime = Field(
        default_factory=datetime.utcnow,
        index=True,
        description="UTC timestamp of capture"
    )
    payload: dict = Field(
        sa_column=Column(JSON),
        description="Progress data: generated_at, league_mode, tracked_leagues_count, progress, budget_subset"
    )
    source: str = Field(
        default="dashboard_manual",
        max_length=50,
        description="Origin: dashboard_manual, scheduler_daily, script"
    )
    app_commit: Optional[str] = Field(
        default=None,
        max_length=40,
        description="Git SHA if available via env"
    )


class UnmappedEntityBacklog(SQLModel, table=True):
    """
    Backlog of unmapped entities from data providers.

    Tracks entities (teams, leagues, matches) that couldn't be mapped
    to our internal IDs during ingestion. Allows manual resolution
    and measures mapping coverage.

    Part of Data Quality Telemetry P0.4.
    """

    __tablename__ = "unmapped_entities_backlog"
    __table_args__ = (
        UniqueConstraint("provider", "entity_type", "raw_key", name="uq_unmapped_entity"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    provider: str = Field(
        max_length=50,
        index=True,
        description="Data provider (e.g., 'api_football', 'football_data_uk')"
    )
    entity_type: str = Field(
        max_length=50,
        index=True,
        description="Entity type: 'team', 'league', 'match', 'player'"
    )
    raw_key: str = Field(
        max_length=255,
        description="Raw identifier from provider (ID or name)"
    )
    raw_name: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Human-readable name if available"
    )
    league_context: Optional[str] = Field(
        default=None,
        max_length=255,
        description="League/competition context for resolution"
    )
    first_seen_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this unmapped entity was first encountered"
    )
    last_seen_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this unmapped entity was last seen"
    )
    count_seen: int = Field(
        default=1,
        description="How many times this entity was encountered"
    )
    resolved: bool = Field(
        default=False,
        index=True,
        description="Whether this entity has been resolved/mapped"
    )
    resolved_to_id: Optional[int] = Field(
        default=None,
        description="Internal ID it was resolved to (if resolved)"
    )
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="When it was resolved"
    )


class LeagueStandings(SQLModel, table=True):
    """
    Persisted league standings for DB-first architecture.

    Standings are fetched from API-Football and stored in DB.
    The /matches/{id}/details endpoint serves from DB; provider is fallback only.
    Refresh job updates standings every 6-24h per league.
    """

    __tablename__ = "league_standings"
    __table_args__ = (
        UniqueConstraint("league_id", "season", name="uq_league_season"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    league_id: int = Field(index=True, description="API-Football league ID")
    season: int = Field(index=True, description="Season year (e.g., 2024)")

    # Standings payload (list of team standings)
    standings: list = Field(
        sa_column=Column(JSON),
        description="Full standings array from API-Football"
    )

    # Metadata
    captured_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When standings were fetched from provider"
    )
    source: str = Field(
        default="api_football",
        max_length=50,
        description="Data source (api_football, manual, etc.)"
    )

    # Cache control
    expires_at: Optional[datetime] = Field(
        default=None,
        description="When this data should be refreshed (TTL)"
    )


class LeagueSeasonBaseline(SQLModel, table=True):
    """
    Aggregated baselines per league/season for narrative context.

    Provides league-wide averages and percentages for goals, cards, corners, etc.
    Used in derived_facts to give narratives relative context.
    """

    __tablename__ = "league_season_baselines"
    __table_args__ = (
        UniqueConstraint("league_id", "season", "as_of_date", name="uq_baseline_league_season_date"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    league_id: int = Field(index=True, description="API-Football league ID")
    season: int = Field(index=True, description="Season year (e.g., 2025)")
    as_of_date: datetime = Field(index=True, description="Date this baseline represents")

    # Sample size
    sample_n_matches: int = Field(description="Number of matches used for calculation")

    # Goals metrics (P0)
    goals_avg_per_match: float = Field(description="Average total goals per match")
    over_1_5_pct: float = Field(description="Percentage of matches with >1.5 goals")
    over_2_5_pct: float = Field(description="Percentage of matches with >2.5 goals")
    over_3_5_pct: float = Field(description="Percentage of matches with >3.5 goals")
    btts_yes_pct: float = Field(description="Both teams to score percentage")

    # Clean sheet / failed to score (P0)
    clean_sheet_pct_home: float = Field(description="Home team clean sheet percentage")
    clean_sheet_pct_away: float = Field(description="Away team clean sheet percentage")
    failed_to_score_pct_home: float = Field(description="Home team failed to score percentage")
    failed_to_score_pct_away: float = Field(description="Away team failed to score percentage")

    # Cards and corners (nullable - depends on data availability)
    corners_avg_per_match: Optional[float] = Field(default=None, description="Average corners per match")
    yellow_cards_avg_per_match: Optional[float] = Field(default=None, description="Average yellow cards per match")
    red_cards_avg_per_match: Optional[float] = Field(default=None, description="Average red cards per match")

    # Metadata
    last_computed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this baseline was computed"
    )


class LeagueTeamProfile(SQLModel, table=True):
    """
    Team-level aggregated profile within a league/season.

    Provides team-specific rates, percentages, and ranks relative to the league.
    Used in derived_facts for team context in narratives.
    """

    __tablename__ = "league_team_profiles"
    __table_args__ = (
        UniqueConstraint("league_id", "season", "team_id", "as_of_date", name="uq_profile_league_team_date"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    league_id: int = Field(index=True, description="API-Football league ID")
    season: int = Field(index=True, description="Season year (e.g., 2025)")
    team_id: int = Field(index=True, foreign_key="teams.id", description="Internal team ID")
    as_of_date: datetime = Field(index=True, description="Date this profile represents")

    # Sample size and confidence
    matches_played: int = Field(description="Matches played in this league/season")
    min_sample_ok: bool = Field(default=False, description="True if sample >= min threshold (5)")
    rank_confidence: str = Field(default="low", max_length=10, description="'low' (<10 matches) or 'high' (>=10)")

    # Goals metrics (P0.2)
    goals_for_per_match: float = Field(description="Goals scored per match")
    goals_against_per_match: float = Field(description="Goals conceded per match")
    goal_difference_per_match: float = Field(description="Goal difference per match")

    # Percentages (P0.2)
    clean_sheet_pct: float = Field(description="Clean sheet percentage")
    failed_to_score_pct: float = Field(description="Failed to score percentage")
    btts_yes_pct: float = Field(description="Both teams scored percentage")
    over_1_5_pct: float = Field(description="Over 1.5 goals percentage")
    over_2_5_pct: float = Field(description="Over 2.5 goals percentage")
    over_3_5_pct: float = Field(description="Over 3.5 goals percentage")

    # Cards and corners (nullable)
    corners_for_per_match: Optional[float] = Field(default=None, description="Corners won per match")
    corners_against_per_match: Optional[float] = Field(default=None, description="Corners conceded per match")
    yellow_cards_per_match: Optional[float] = Field(default=None, description="Yellow cards per match")
    red_cards_per_match: Optional[float] = Field(default=None, description="Red cards per match")

    # Ranks within league (P0.3) - nullable if min_sample_ok=False
    rank_best_attack: Optional[int] = Field(default=None, description="Rank by goals_for (1=best)")
    rank_worst_defense: Optional[int] = Field(default=None, description="Rank by goals_against (1=worst)")
    rank_goal_difference: Optional[int] = Field(default=None, description="Rank by goal difference (1=best)")
    rank_most_corners: Optional[int] = Field(default=None, description="Rank by corners won (1=most)")
    rank_most_cards: Optional[int] = Field(default=None, description="Rank by cards received (1=most)")
    total_teams_in_league: Optional[int] = Field(default=None, description="Total teams for rank context")

    # By-time goals (P1)
    goals_scored_0_15_pct: Optional[float] = Field(default=None, description="% of goals scored in 0-15 min")
    goals_scored_76_90p_pct: Optional[float] = Field(default=None, description="% of goals scored in 76-90+ min")
    goals_conceded_0_15_pct: Optional[float] = Field(default=None, description="% of goals conceded in 0-15 min")
    goals_conceded_76_90p_pct: Optional[float] = Field(default=None, description="% of goals conceded in 76-90+ min")

    # Metadata
    last_computed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this profile was computed"
    )


class ShadowPrediction(SQLModel, table=True):
    """
    Shadow predictions for A/B model comparison.

    Stores predictions from both baseline and shadow (experimental) models
    for the same matches. Used for evaluating new model architectures
    in production without affecting served predictions.

    FASE 2: Two-stage model shadow evaluation.
    """

    __tablename__ = "shadow_predictions"
    __table_args__ = (
        UniqueConstraint("match_id", name="uq_shadow_match_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    match_id: int = Field(foreign_key="matches.id", index=True)

    # Baseline model predictions (currently served)
    baseline_version: str = Field(max_length=50, description="e.g., 'v1.0.0'")
    baseline_home_prob: float
    baseline_draw_prob: float
    baseline_away_prob: float
    baseline_predicted: str = Field(max_length=10, description="'home', 'draw', or 'away'")

    # Shadow model predictions (experimental, not served)
    shadow_version: str = Field(max_length=50, description="e.g., 'v1.1.0-twostage'")
    shadow_architecture: str = Field(max_length=50, description="'baseline' or 'two_stage'")
    shadow_home_prob: float
    shadow_draw_prob: float
    shadow_away_prob: float
    shadow_predicted: str = Field(max_length=10, description="'home', 'draw', or 'away'")

    # Outcome (filled after match completes)
    actual_result: Optional[str] = Field(
        default=None, max_length=10, description="'home', 'draw', or 'away'"
    )
    baseline_correct: Optional[bool] = Field(default=None)
    shadow_correct: Optional[bool] = Field(default=None)

    # Metrics (computed after outcome)
    baseline_brier: Optional[float] = Field(default=None, description="Brier contribution")
    shadow_brier: Optional[float] = Field(default=None, description="Brier contribution")

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    evaluated_at: Optional[datetime] = Field(default=None, description="When outcome was recorded")


class SensorPrediction(SQLModel, table=True):
    """
    Sensor B predictions for calibration diagnostics.

    Stores predictions from Model A (production) and Sensor B (sliding-window LogReg)
    for comparison. Used to detect if Model A has become stale/rigid.

    AUDIT P0: Column names aligned with migrations/add_sensor_predictions_table.sql
    """

    __tablename__ = "sensor_predictions"
    __table_args__ = (
        UniqueConstraint("match_id", name="uq_sensor_match_id"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    match_id: int = Field(foreign_key="matches.id", index=True)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    evaluated_at: Optional[datetime] = Field(default=None)
    window_size: int = Field(description="Training window size at prediction time")

    # Model versions
    model_a_version: str = Field(max_length=50, description="Model A version")
    model_b_version: Optional[str] = Field(default=None, max_length=50, description="Sensor version, NULL if LEARNING")

    # Model A predictions (production) - always present
    a_home_prob: float
    a_draw_prob: float
    a_away_prob: float
    a_pick: str = Field(max_length=10, description="'home', 'draw', or 'away'")

    # Sensor B predictions (diagnostic) - NULL if sensor in LEARNING state
    b_home_prob: Optional[float] = Field(default=None)
    b_draw_prob: Optional[float] = Field(default=None)
    b_away_prob: Optional[float] = Field(default=None)
    b_pick: Optional[str] = Field(default=None, max_length=10, description="'home', 'draw', or 'away'")

    # Outcome (filled after match completes)
    actual_outcome: Optional[str] = Field(
        default=None, max_length=10, description="'home', 'draw', or 'away'"
    )
    a_correct: Optional[bool] = Field(default=None)
    b_correct: Optional[bool] = Field(default=None)

    # Metrics (computed after outcome)
    a_brier: Optional[float] = Field(default=None)
    b_brier: Optional[float] = Field(default=None)

    # Sensor state at prediction time
    sensor_state: str = Field(default="LEARNING", max_length=20, description="LEARNING, READY, ERROR")


class OpsAlert(SQLModel, table=True):
    """
    Alerts from Grafana Alerting webhook for ops dashboard notifications.

    Used for bell icon + toast in /dashboard/ops. Grafana is source of truth
    for alert rules; this table just stores/displays them.
    """

    __tablename__ = "ops_alerts"
    __table_args__ = (
        UniqueConstraint("dedupe_key", name="uq_ops_alerts_dedupe_key"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)

    # Deduplication key (Grafana fingerprint or computed)
    dedupe_key: str = Field(max_length=255, index=True, description="Unique key for idempotent upserts")

    # Alert status and severity
    status: str = Field(default="firing", max_length=20, description="firing or resolved")
    severity: str = Field(default="warning", max_length=20, description="critical, warning, info")

    # Content
    title: str = Field(max_length=500, description="Alert title/summary")
    message: Optional[str] = Field(default=None, description="Alert description (truncated to 1000 chars)")

    # Grafana metadata (stored as JSON)
    labels: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    annotations: Optional[dict] = Field(default=None, sa_column=Column(JSON))

    # Timestamps from Grafana
    starts_at: Optional[datetime] = Field(default=None, description="When alert started firing")
    ends_at: Optional[datetime] = Field(default=None, description="When alert resolved")

    # Source info
    source: str = Field(default="grafana", max_length=50, description="Alert source")
    source_url: Optional[str] = Field(default=None, description="Link to Grafana alert/silence")

    # Tracking
    first_seen_at: datetime = Field(default_factory=datetime.utcnow, description="First time we saw this alert")
    last_seen_at: datetime = Field(default_factory=datetime.utcnow, description="Last webhook received")

    # User interaction
    is_read: bool = Field(default=False, description="User has seen in bell dropdown")
    is_ack: bool = Field(default=False, description="User has acknowledged")

    # Standard timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class OpsIncident(SQLModel, table=True):
    """
    Persistent incidents from _aggregate_incidents().

    Sources: sentry, predictions, jobs, fastpath, budget.
    Supports acknowledge/resolve with timeline tracking.
    Auto-resolve with grace window to avoid flapping.
    """

    __tablename__ = "ops_incidents"
    __table_args__ = (
        UniqueConstraint("source", "source_key", name="uq_ops_incidents_source_key"),
    )

    # Stable ID from make_id() hash
    id: int = Field(primary_key=True, description="MD5 hash first 8 hex as int")

    # Source identification
    source: str = Field(max_length=30, index=True, description="sentry|predictions|jobs|fastpath|budget")
    source_key: str = Field(max_length=100, description="Key within source for dedup")

    # Severity and status
    severity: str = Field(max_length=20, description="critical|warning|info")
    status: str = Field(default="active", max_length=20, description="active|acknowledged|resolved")
    type: str = Field(max_length=30, description="Backend type: sentry|predictions|scheduler|llm|api_budget")

    # Content
    title: str = Field(max_length=200)
    description: Optional[str] = Field(default=None)
    details: Optional[dict] = Field(default=None, sa_column=Column(JSON))
    runbook_url: Optional[str] = Field(default=None, max_length=500)

    # Timeline: [{ts, message, actor, action}]
    timeline: list = Field(default_factory=list, sa_column=Column(JSON, nullable=False, server_default="[]"))

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="First detection (never overwritten)")
    last_seen_at: datetime = Field(default_factory=datetime.utcnow, description="Last time source reported")
    acknowledged_at: Optional[datetime] = Field(default=None)
    resolved_at: Optional[datetime] = Field(default=None)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TeamOverride(SQLModel, table=True):
    """
    Team identity overrides for display purposes.

    Handles rebranding cases where API-Football hasn't updated team names/logos
    but we need to show the correct identity to users. Preserves historical data
    while showing correct names for current/future matches.

    Example: La Equidad → Internacional de Bogotá (2026-01-01)
    """

    __tablename__ = "team_overrides"
    __table_args__ = (
        UniqueConstraint("provider", "external_team_id", "effective_from", name="uq_team_override"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    provider: str = Field(
        max_length=50,
        default="api_football",
        description="Data provider (api_football, etc.)"
    )
    external_team_id: int = Field(
        index=True,
        description="Provider's team ID (e.g., API-Football team ID)"
    )

    # Display overrides
    display_name: str = Field(
        max_length=255,
        description="Name to display instead of provider's name"
    )
    display_logo_url: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Logo URL to display instead of provider's logo"
    )

    # Temporal validity
    effective_from: datetime = Field(
        index=True,
        description="Override applies to matches on or after this date"
    )
    effective_to: Optional[datetime] = Field(
        default=None,
        description="Override applies to matches before this date (null = indefinite)"
    )

    # Metadata
    reason: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Why this override exists (rebranding, merger, etc.)"
    )
    updated_by: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Who created/updated this override"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class JobRun(SQLModel, table=True):
    """
    Scheduler job execution tracking.

    Persists job runs for ops dashboard fallback when Prometheus metrics
    are unavailable (e.g., cold-start after deploy). Enables jobs_health
    to show last_success_at from DB instead of "unknown".

    Jobs tracked: stats_backfill, odds_sync, fastpath, daily_save, etc.
    """

    __tablename__ = "job_runs"

    id: Optional[int] = Field(default=None, primary_key=True)
    job_name: str = Field(max_length=100, index=True, description="Job identifier")
    status: str = Field(
        max_length=20,
        default="ok",
        description="ok, error, rate_limited, budget_exceeded"
    )
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = Field(default=None)
    duration_ms: Optional[int] = Field(default=None)
    error_message: Optional[str] = Field(default=None, description="Error details if failed")
    metrics: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Job-specific metrics (rows_updated, fixtures_scanned, etc.)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


class PredictionRerun(SQLModel, table=True):
    """
    Audit log for manual prediction reruns.

    Tracks before/after stats when re-predicting NS matches with a different
    model architecture. Used for controlled model promotion and A/B analysis.

    The is_active flag controls serving preference:
    - True: serve predictions from this rerun (two-stage)
    - False: rollback to baseline (serve baseline predictions)
    """

    __tablename__ = "prediction_reruns"

    id: Optional[int] = Field(default=None, primary_key=True)
    run_id: UUID = Field(
        sa_column=Column(PG_UUID(as_uuid=True), nullable=False, unique=True),
        description="Unique identifier for this rerun"
    )
    run_type: str = Field(
        max_length=50,
        description="'manual_rerun', 'model_promotion', 'rollback'"
    )

    # Configuration
    window_hours: int = Field(description="Time window for NS matches")
    architecture_before: str = Field(max_length=50, description="e.g., 'baseline'")
    architecture_after: str = Field(max_length=50, description="e.g., 'two_stage'")
    model_version_before: str = Field(max_length=50, description="e.g., 'v1.0.0'")
    model_version_after: str = Field(max_length=50, description="e.g., 'v1.1.0-twostage'")

    # Scope
    matches_total: int = Field(description="Total NS matches in window")
    matches_with_odds: int = Field(description="Matches with odds coverage")

    # Before/After stats (JSON)
    stats_before: dict = Field(
        sa_column=Column(JSON, nullable=False),
        description="Stats snapshot before rerun"
    )
    stats_after: dict = Field(
        sa_column=Column(JSON, nullable=False),
        description="Stats snapshot after rerun"
    )

    # Top changes for review
    top_deltas: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Top N matches with largest probability changes"
    )

    # Outcome metrics (filled when matches complete)
    evaluation_window_days: int = Field(default=14, description="Days to wait for evaluation")
    evaluated_matches: int = Field(default=0, description="Matches evaluated so far")
    evaluation_report: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Accuracy/Brier comparison after matches complete"
    )

    # Status for serving preference (rollback without deleting)
    is_active: bool = Field(
        default=True,
        description="If False, rollback: serve baseline instead of rerun predictions"
    )

    # Metadata
    triggered_by: Optional[str] = Field(default=None, max_length=100)
    notes: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    evaluated_at: Optional[datetime] = Field(default=None)


class OpsAuditLog(SQLModel, table=True):
    """
    Audit log for OPS dashboard actions.

    Tracks manual triggers, syncs, and other operational actions
    for accountability and debugging.
    """

    __tablename__ = "ops_audit_log"

    id: Optional[int] = Field(default=None, primary_key=True)

    # Action identification
    action: str = Field(
        max_length=100,
        index=True,
        description="Action type: predictions_trigger, odds_sync, sync_window, etc."
    )
    request_id: str = Field(
        max_length=36,
        description="UUID for request correlation"
    )

    # Actor identification
    actor: str = Field(
        max_length=100,
        description="Actor type: dashboard_token, scheduler, api_key"
    )
    actor_id: str = Field(
        max_length=32,
        index=True,
        description="Short hash of token/session for identification"
    )

    # Request context
    ip_address: Optional[str] = Field(
        default=None,
        max_length=45,
        description="Client IP address (IPv4 or IPv6)"
    )
    user_agent: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Client user agent"
    )

    # Action parameters and result
    params: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Action parameters"
    )
    result: str = Field(
        max_length=20,
        description="Result: ok, error, rejected"
    )
    result_detail: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSON),
        description="Detailed result/response summary"
    )
    error_message: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Error message if result=error"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        index=True
    )
    duration_ms: Optional[int] = Field(
        default=None,
        description="Action duration in milliseconds"
    )


# =============================================================================
# SOTA SOFASCORE MODELS
# =============================================================================


class MatchSofascorePlayer(SQLModel, table=True):
    """
    Sofascore XI player snapshot per match (ARCHITECTURE_SOTA.md 1.3).

    Stores pre-kickoff XI data including player ratings and positions.
    Used for xi_* feature engineering with PIT (captured_at < kickoff).
    """

    __tablename__ = "match_sofascore_player"
    __table_args__ = (
        UniqueConstraint("match_id", "team_side", "player_id_ext", name="pk_sofascore_player"),
    )

    # Composite primary key fields
    match_id: int = Field(foreign_key="matches.id", primary_key=True)
    team_side: str = Field(max_length=10, primary_key=True, description="'home' or 'away'")
    player_id_ext: str = Field(max_length=100, primary_key=True, description="Sofascore player ID")

    # Player data
    position: str = Field(max_length=20, description="GK/DEF/MID/FWD or sub-role")
    is_starter: bool = Field(description="True if in starting XI")
    rating_pre_match: Optional[float] = Field(default=None, description="Pre-match rating if available")
    rating_recent_form: Optional[float] = Field(default=None, description="Recent form rating (rolling avg)")
    minutes_expected: Optional[int] = Field(default=None, description="Expected minutes if available")

    # Point-in-time tracking
    captured_at: datetime = Field(default_factory=datetime.utcnow, description="When snapshot was captured (UTC)")


class MatchSofascoreLineup(SQLModel, table=True):
    """
    Sofascore formation snapshot per match (ARCHITECTURE_SOTA.md 1.3).

    Stores pre-kickoff team formation (e.g., 4-3-3, 4-2-3-1).
    Used for formation_* features with PIT (captured_at < kickoff).
    """

    __tablename__ = "match_sofascore_lineup"
    __table_args__ = (
        UniqueConstraint("match_id", "team_side", name="pk_sofascore_lineup"),
    )

    # Composite primary key fields
    match_id: int = Field(foreign_key="matches.id", primary_key=True)
    team_side: str = Field(max_length=10, primary_key=True, description="'home' or 'away'")

    # Formation data
    formation: str = Field(max_length=20, description="Formation string, e.g., '4-3-3', '4-2-3-1'")

    # Point-in-time tracking
    captured_at: datetime = Field(default_factory=datetime.utcnow, description="When snapshot was captured (UTC)")


class Player(SQLModel, table=True):
    """
    Player catalog from API-Football squads endpoint.

    Used to cross-reference with match_lineups.starting_xi_ids for XI continuity.
    external_id is the API-Football player ID (same as in lineups and injuries).
    """

    __tablename__ = "players"

    id: Optional[int] = Field(default=None, primary_key=True)
    external_id: int = Field(unique=True, index=True, description="API-Football player ID")
    name: str = Field(max_length=200, description="Player full name")
    position: Optional[str] = Field(default=None, max_length=20, description="Goalkeeper/Defender/Midfielder/Attacker")
    team_id: Optional[int] = Field(default=None, foreign_key="teams.id", description="Current team (internal ID)")
    team_external_id: Optional[int] = Field(default=None, description="Current team (API-Football ID)")
    jersey_number: Optional[int] = Field(default=None, description="Squad number")
    age: Optional[int] = Field(default=None, description="Current age from API")
    photo_url: Optional[str] = Field(default=None, max_length=500, description="Player headshot URL")
    firstname: Optional[str] = Field(default=None, max_length=100)
    lastname: Optional[str] = Field(default=None, max_length=150)
    birth_date: Optional[date] = Field(default=None, description="Birth date YYYY-MM-DD")
    birth_place: Optional[str] = Field(default=None, max_length=200)
    birth_country: Optional[str] = Field(default=None, max_length=100)
    nationality: Optional[str] = Field(default=None, max_length=100)
    height: Optional[str] = Field(default=None, max_length=10, description="Height in cm")
    weight: Optional[str] = Field(default=None, max_length=10, description="Weight in kg")
    last_synced_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Last sync timestamp")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, description="Row creation timestamp")


class PlayerPhotoAsset(SQLModel, table=True):
    """HQ player photo asset stored in R2.

    Supports global headshots (context_team_id=NULL) and
    contextual composed cards (context_team_id=team_id).
    Immutable R2 keys via content_hash prefix.
    """

    __tablename__ = "player_photo_assets"

    id: Optional[int] = Field(default=None, primary_key=True)
    player_external_id: int = Field(index=True, description="API-Football player ID")
    context_team_id: Optional[int] = Field(default=None, description="NULL=global headshot, team_id=contextual")
    season: Optional[str] = Field(default=None, max_length=20, description="e.g. 2025, 2025-2026")
    role: Optional[str] = Field(default=None, max_length=10, description="field | gk")
    kit_variant: Optional[str] = Field(default=None, max_length=10, description="home | away | third")

    # Asset dimensions (fix #3: separate asset_type + style)
    asset_type: str = Field(max_length=10, description="card | thumb")
    style: str = Field(default="raw", max_length=20, description="raw | segmented | composed")

    # Storage (fix #1: content_hash in key for immutability)
    r2_key: str = Field(max_length=500, description="R2 object key")
    cdn_url: str = Field(max_length=500, description="Full CDN URL")
    content_hash: str = Field(max_length=64, description="SHA-256 hex digest")
    revision: int = Field(default=1)

    # Provenance
    source: str = Field(max_length=50, description="sofascore | api_football | wikimedia | club_site")
    processor: Optional[str] = Field(default=None, max_length=50, description="photoroom | rembg | none")
    quality_score: Optional[int] = Field(default=None, description="0-100 identity + vision score")
    photo_meta: Optional[dict] = Field(default=None, sa_column=Column(JSONB))

    # Lifecycle
    review_status: str = Field(default="pending_review", max_length=20, description="pending_review | approved | rejected | superseded")
    is_active: bool = Field(default=False)
    activated_at: Optional[datetime] = Field(default=None)
    deactivated_at: Optional[datetime] = Field(default=None)
    changed_by: Optional[str] = Field(default=None, max_length=20, description="pipeline | manual | rollback")
    run_id: Optional[str] = Field(default=None, max_length=36)

    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


class MatchLineup(SQLModel, table=True):
    """
    API-Football lineup per match/team.

    Stores starting XI IDs, substitute IDs, formation, and coach.
    starting_xi_ids are API-Football player IDs that cross-reference with players.external_id.
    """

    __tablename__ = "match_lineups"

    id: Optional[int] = Field(default=None, primary_key=True)
    match_id: int = Field(foreign_key="matches.id", description="Internal match ID")
    team_id: int = Field(foreign_key="teams.id", description="Internal team ID")
    is_home: bool = Field(description="True if home team")
    formation: Optional[str] = Field(default=None, max_length=20, description="e.g. 4-3-3")
    starting_xi_ids: list = Field(sa_column=Column(JSON), description="API-Football player IDs [int]")
    starting_xi_names: list = Field(sa_column=Column(JSON), description="Player names [str]")
    starting_xi_positions: list = Field(sa_column=Column(JSON), description="Player positions [str]")
    substitutes_ids: list = Field(sa_column=Column(JSON), description="Substitute player IDs [int]")
    substitutes_names: list = Field(sa_column=Column(JSON), description="Substitute names [str]")
    coach_id: Optional[int] = Field(default=None, description="Coach API-Football ID")
    coach_name: Optional[str] = Field(default=None, max_length=200)
    lineup_confirmed_at: Optional[datetime] = Field(default=None, description="When lineup was confirmed")
    source: Optional[str] = Field(default=None, max_length=50, description="e.g. api-football")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)


class OpsSetting(SQLModel, table=True):
    """
    Dynamic configuration settings for ops dashboard.

    Key-value store with JSONB values for flexible configuration.
    Used for IA Features settings, feature flags, and other runtime config.
    """

    __tablename__ = "ops_settings"

    key: str = Field(max_length=100, primary_key=True, description="Setting identifier (e.g., 'ia_features')")
    value: dict = Field(sa_column=Column(JSONB, nullable=False), description="Configuration as JSON object")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp (UTC)")
    updated_by: str = Field(max_length=50, default="system", description="Who updated (user email or 'system')")


# =============================================================================
# 3D LOGO GENERATION SYSTEM
# =============================================================================


class TeamLogo(SQLModel, table=True):
    """
    3D logo variants for teams/national teams.

    Stores R2 keys for original and 3 generated variants:
    - original: uploaded source image
    - front_3d: frontal 3D metallic badge
    - facing_right: HOME perspective (looks at opponent on right)
    - facing_left: AWAY perspective (looks at opponent on left)

    Spec: docs/TEAM_LOGOS_3D_SPEC.md
    """

    __tablename__ = "team_logos"

    team_id: int = Field(
        foreign_key="teams.id",
        primary_key=True,
        description="FK to teams.id"
    )

    # R2 Storage Keys
    r2_key_original: Optional[str] = Field(default=None, max_length=255)
    r2_key_original_svg: Optional[str] = Field(default=None, max_length=255)  # Original SVG preserved
    r2_key_front: Optional[str] = Field(default=None, max_length=255)
    r2_key_right: Optional[str] = Field(default=None, max_length=255)
    r2_key_left: Optional[str] = Field(default=None, max_length=255)

    # URLs de thumbnails (generadas post-resize)
    urls: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSONB, server_default="{}"),
        description="Thumbnail URLs: {front: {64, 128, 256, 512}, right: {...}, left: {...}}"
    )

    # Fallback (API-Football URL original)
    fallback_url: Optional[str] = Field(default=None, max_length=500)

    # Pipeline Status
    status: str = Field(
        default="pending",
        max_length=20,
        description="pending|queued|processing|pending_resize|ready|error|paused"
    )

    # Processing Metadata
    batch_job_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), nullable=True)
    )
    generation_mode: Optional[str] = Field(default=None, max_length=20)
    ia_model: Optional[str] = Field(default=None, max_length=50)
    ia_prompt_version: Optional[str] = Field(default=None, max_length=20)
    use_original_as_front: bool = Field(default=False)

    # Immutable versioning (ABE recommendation)
    revision: int = Field(
        default=1,
        description="Asset revision number, increments on regeneration for cache busting"
    )

    # Timestamps
    uploaded_at: Optional[datetime] = Field(default=None)
    processing_started_at: Optional[datetime] = Field(default=None)
    processing_completed_at: Optional[datetime] = Field(default=None)
    resize_completed_at: Optional[datetime] = Field(default=None)

    # Cost tracking
    ia_cost_usd: Optional[float] = Field(default=None, description="Total IA cost")

    # Error Handling
    error_message: Optional[str] = Field(default=None)
    error_phase: Optional[str] = Field(default=None, max_length=20)
    retry_count: int = Field(default=0)
    last_retry_at: Optional[datetime] = Field(default=None)

    # Validation
    validation_errors: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    last_validation_at: Optional[datetime] = Field(default=None)

    # Review (Liga-by-liga approval)
    review_status: str = Field(default="pending", max_length=20)
    review_notes: Optional[str] = Field(default=None)
    reviewed_by: Optional[str] = Field(default=None, max_length=100)
    reviewed_at: Optional[datetime] = Field(default=None)

    # Audit
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class CompetitionLogo(SQLModel, table=True):
    """
    3D logos for leagues/tournaments (main variant only, no facing).

    Spec: docs/TEAM_LOGOS_3D_SPEC.md
    """

    __tablename__ = "competition_logos"

    league_id: int = Field(primary_key=True, description="FK to admin_leagues.league_id")

    # R2 Storage Keys
    r2_key_original: Optional[str] = Field(default=None, max_length=255)
    r2_key_main: Optional[str] = Field(default=None, max_length=255)

    # URLs de thumbnails
    urls: Optional[dict] = Field(
        default=None,
        sa_column=Column(JSONB, server_default="{}"),
        description="Thumbnail URLs: {64, 128, 256, 512}"
    )

    # Fallback
    fallback_url: Optional[str] = Field(default=None, max_length=500)

    # Status
    status: str = Field(
        default="pending",
        max_length=20,
        description="pending|queued|processing|pending_resize|ready|error"
    )

    # Metadata
    batch_job_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), nullable=True)
    )
    ia_model: Optional[str] = Field(default=None, max_length=50)
    ia_prompt_version: Optional[str] = Field(default=None, max_length=20)
    ia_cost_usd: Optional[float] = Field(default=None)

    # Immutable versioning (ABE recommendation)
    revision: int = Field(
        default=1,
        description="Asset revision number, increments on regeneration for cache busting"
    )

    # Error handling
    error_message: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0)

    # Validation
    validation_errors: Optional[dict] = Field(default=None, sa_column=Column(JSONB))
    last_validation_at: Optional[datetime] = Field(default=None)

    # Timestamps
    uploaded_at: Optional[datetime] = Field(default=None)
    processing_completed_at: Optional[datetime] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class LogoBatchJob(SQLModel, table=True):
    """
    Batch generation jobs with pause/resume/cancel support.

    Tracks liga-by-liga generation progress and approval workflow.

    Spec: docs/TEAM_LOGOS_3D_SPEC.md
    """

    __tablename__ = "logo_batch_jobs"

    id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, server_default="gen_random_uuid()")
    )

    # Configuration
    ia_model: str = Field(max_length=50)
    generation_mode: str = Field(default="full_3d", max_length=20)
    prompt_front: Optional[str] = Field(default=None)
    prompt_right: Optional[str] = Field(default=None)
    prompt_left: Optional[str] = Field(default=None)
    prompt_version: str = Field(max_length=20)

    # Scope
    entity_type: str = Field(default="league", max_length=20)
    league_id: Optional[int] = Field(default=None)
    total_teams: int
    team_ids: Optional[list] = Field(default=None, sa_column=Column(JSON))

    # Status
    status: str = Field(
        default="running",
        max_length=20,
        description="running|paused|completed|cancelled|error|pending_review"
    )

    # Progress
    processed_teams: int = Field(default=0)
    processed_images: int = Field(default=0)
    failed_teams: int = Field(default=0)

    # Cost
    estimated_cost_usd: Optional[float] = Field(default=None)
    actual_cost_usd: float = Field(default=0)

    # Approval
    approval_status: str = Field(default="pending_review", max_length=20)
    approved_count: int = Field(default=0)
    rejected_count: int = Field(default=0)
    approved_by: Optional[str] = Field(default=None, max_length=100)
    approved_at: Optional[datetime] = Field(default=None)

    # Re-run support
    parent_batch_id: Optional[UUID] = Field(
        default=None,
        sa_column=Column(PG_UUID(as_uuid=True), nullable=True)
    )
    is_rerun: bool = Field(default=False)
    rerun_reason: Optional[str] = Field(default=None, max_length=100)

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    paused_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    # Metadata
    started_by: Optional[str] = Field(default=None, max_length=100)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class LogoPromptTemplate(SQLModel, table=True):
    """
    Versioned prompts for A/B testing and rollback support.

    Spec: docs/TEAM_LOGOS_3D_SPEC.md (Kimi consideration)
    """

    __tablename__ = "logo_prompt_templates"
    __table_args__ = (
        UniqueConstraint("version", "variant", "ia_model", name="uq_prompt_version_variant_model"),
    )

    id: Optional[int] = Field(default=None, primary_key=True)
    version: str = Field(max_length=10)
    variant: str = Field(max_length=20, description="front|right|left|main")
    prompt_template: str
    ia_model: Optional[str] = Field(default=None, max_length=50)
    is_active: bool = Field(default=False)
    success_rate: Optional[float] = Field(default=None)
    avg_quality_score: Optional[float] = Field(default=None)
    usage_count: int = Field(default=0)
    notes: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    created_by: Optional[str] = Field(default=None, max_length=100)
