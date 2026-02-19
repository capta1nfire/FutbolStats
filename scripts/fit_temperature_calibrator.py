"""
Fit temperature calibration parameter T from historical predictions.

Joins predictions with prediction_outcomes to get real labels.
Default: --only-frozen (T represents pre-match flow, not cascade).

Usage:
    source .env
    python scripts/fit_temperature_calibrator.py --train-end 2026-01-01 --eval-start 2026-01-01

Output:
    T=X.XX, Brier/LogLoss pre/post, recommended env vars.

ABE P0 (2026-02-19): Train/test split mandatory — no leakage.
"""

import argparse
import asyncio
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    parser = argparse.ArgumentParser(description="Fit temperature calibrator")
    parser.add_argument("--train-end", required=True, help="Train set cutoff (YYYY-MM-DD)")
    parser.add_argument("--eval-start", required=True, help="Eval set start (YYYY-MM-DD)")
    parser.add_argument("--model-version", default=None, help="Filter by model_version")
    parser.add_argument("--only-frozen", action="store_true", default=True,
                        help="Only use frozen (pre-match) predictions (default: True)")
    parser.add_argument("--include-all", action="store_true",
                        help="Include non-frozen predictions too")
    parser.add_argument("--dry-run", action="store_true", help="Print query only")
    args = parser.parse_args()

    if args.include_all:
        args.only_frozen = False

    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy import text

    db_url = os.environ.get("DATABASE_URL_ASYNC") or os.environ.get("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL_ASYNC or DATABASE_URL not set. Run: source .env")
        sys.exit(1)

    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    engine = create_async_engine(db_url, pool_size=2)
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Build query
    frozen_clause = "AND p.is_frozen = true" if args.only_frozen else ""
    version_clause = f"AND p.model_version = '{args.model_version}'" if args.model_version else ""

    query = f"""
        SELECT
            p.home_prob, p.draw_prob, p.away_prob,
            po.actual_result,
            m.date AS match_date
        FROM predictions p
        JOIN prediction_outcomes po ON po.prediction_id = p.id
        JOIN matches m ON m.id = p.match_id
        WHERE po.actual_result IN ('home', 'draw', 'away')
        AND p.home_prob IS NOT NULL
        AND p.draw_prob IS NOT NULL
        AND p.away_prob IS NOT NULL
        {frozen_clause}
        {version_clause}
        ORDER BY m.date
    """

    if args.dry_run:
        print("=== SQL ===")
        print(query)
        await engine.dispose()
        return

    async with async_session() as session:
        result = await session.execute(text(query))
        rows = result.fetchall()

    await engine.dispose()

    if not rows:
        print("ERROR: No data found. Check filters.")
        sys.exit(1)

    print(f"Total rows: {len(rows)}")

    # Split into train/eval
    from datetime import datetime
    train_end = datetime.strptime(args.train_end, "%Y-%m-%d")
    eval_start = datetime.strptime(args.eval_start, "%Y-%m-%d")

    train_probs, train_outcomes = [], []
    eval_probs, eval_outcomes = [], []

    outcome_map = {"home": 0, "draw": 1, "away": 2}

    for row in rows:
        p = [row.home_prob, row.draw_prob, row.away_prob]
        o = outcome_map[row.actual_result]
        md = row.match_date.replace(tzinfo=None) if hasattr(row.match_date, 'replace') else row.match_date

        if md < train_end:
            train_probs.append(p)
            train_outcomes.append(o)
        if md >= eval_start:
            eval_probs.append(p)
            eval_outcomes.append(o)

    train_probs = np.array(train_probs)
    train_outcomes = np.array(train_outcomes)
    eval_probs = np.array(eval_probs)
    eval_outcomes = np.array(eval_outcomes)

    print(f"Train: {len(train_probs)} | Eval: {len(eval_probs)}")

    if len(train_probs) < 50:
        print("WARNING: Train set < 50 samples. Results unreliable.")
    if len(eval_probs) < 50:
        print("WARNING: Eval set < 50 samples. Results unreliable.")

    # Fit temperature
    from app.ml.calibration import TemperatureScaling

    cal = TemperatureScaling(t_min=0.5, t_max=3.0, n_grid=500)
    cal.fit(train_probs, train_outcomes)
    T = cal.temperature

    print(f"\n{'='*50}")
    print(f"Fitted Temperature: T = {T:.4f}")
    print(f"{'='*50}")

    # Metrics
    def brier_score(probs, outcomes):
        n = len(outcomes)
        bs = 0.0
        for i in range(n):
            for c in range(3):
                actual = 1.0 if outcomes[i] == c else 0.0
                bs += (probs[i, c] - actual) ** 2
        return bs / n

    def log_loss(probs, outcomes):
        eps = 1e-10
        ll = 0.0
        for i in range(len(outcomes)):
            ll -= np.log(max(probs[i, outcomes[i]], eps))
        return ll / len(outcomes)

    # Eval pre/post calibration
    cal_eval = cal.transform(eval_probs)

    brier_pre = brier_score(eval_probs, eval_outcomes)
    brier_post = brier_score(cal_eval, eval_outcomes)
    ll_pre = log_loss(eval_probs, eval_outcomes)
    ll_post = log_loss(cal_eval, eval_outcomes)

    print(f"\n--- Eval Set (N={len(eval_probs)}) ---")
    print(f"Brier:   {brier_pre:.6f} → {brier_post:.6f} (Δ={brier_post - brier_pre:+.6f})")
    print(f"LogLoss: {ll_pre:.6f} → {ll_post:.6f} (Δ={ll_post - ll_pre:+.6f})")

    if brier_post > brier_pre:
        print("\nWARNING: Calibration WORSENED Brier on eval set. T=1.0 (no-op) recommended.")

    print(f"\n--- Recommended env vars ---")
    print(f"PROBA_CALIBRATION_ENABLED=true")
    print(f"PROBA_CALIBRATION_METHOD=temperature")
    print(f"PROBA_CALIBRATION_TEMPERATURE={T:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
