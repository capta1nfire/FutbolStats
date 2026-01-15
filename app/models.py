"""Database models using SQLModel."""

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column, LargeBinary, UniqueConstraint
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
    events: Optional[list] = Field(
        default=None, sa_column=Column(JSON), description="Match events (goals, cards, etc.)"
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
    odds_recorded_at: Optional[datetime] = Field(default=None, description="When odds were last recorded")

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
        UniqueConstraint("match_id", "created_at", name="uq_shadow_match_created"),
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
