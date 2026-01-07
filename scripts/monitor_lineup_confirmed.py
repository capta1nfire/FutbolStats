#!/usr/bin/env python3
"""
Monitor Semanal de lineup_confirmed - Auditoría PIT

Este script genera reportes semanales de captura de lineup_confirmed
para preparar evaluación PIT (Point-In-Time) del modelo.

Requisitos del asesor:
1) Reporte semanal con:
   - # lineup_confirmed snapshots capturados esa semana
   - % odds_freshness=live vs stale
   - distribución delta_to_kickoff_seconds (p50/p90)
   - cobertura por liga y por bookmaker
2) Gating/threshold "principal" congelados:
   - gate actual + threshold 0.02 + stake fijo
3) Trigger de evaluación PIT:
   - N >= 200: evaluación PIT preliminar
   - N >= 500: evaluación PIT completa
   - Decisión: CONTINUE/CLOSE/INCONCLUSIVE con CI95%

Run weekly:
    DATABASE_URL="postgresql://..." python scripts/monitor_lineup_confirmed.py

Output: JSON report + console summary
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = os.environ.get("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")

if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")


# =============================================================================
# FROZEN CONFIGURATION (DO NOT MODIFY - Pre-registered)
# =============================================================================
FROZEN_CONFIG = {
    "edge_threshold": 0.02,
    "stake": 1.0,  # Fixed stake
    "use_gate": True,  # Gate actual (lineup surprise, missing starters)
    "odds_type": "opening",
}

# PIT Milestones
MILESTONE_PRELIMINARY = 200
MILESTONE_FULL = 500


# =============================================================================
# DATA COLLECTION
# =============================================================================

async def get_weekly_stats(engine, weeks_back: int = 1) -> dict:
    """Get lineup_confirmed stats for the past N weeks."""
    async with engine.connect() as conn:
        # Total REAL lineup_confirmed (not simulated)
        result = await conn.execute(text("""
            SELECT COUNT(*)
            FROM odds_snapshots
            WHERE snapshot_type = 'lineup_confirmed'
        """))
        total_real = result.scalar() or 0

        # Total with live odds (valid for PIT)
        result = await conn.execute(text("""
            SELECT COUNT(*)
            FROM odds_snapshots
            WHERE snapshot_type = 'lineup_confirmed'
              AND odds_freshness = 'live'
              AND delta_to_kickoff_seconds > 0
        """))
        total_live_valid = result.scalar() or 0

        # This week's captures (using interval syntax that works with asyncpg)
        days = weeks_back * 7
        result = await conn.execute(text(f"""
            SELECT COUNT(*)
            FROM odds_snapshots
            WHERE snapshot_type = 'lineup_confirmed'
              AND snapshot_at > NOW() - INTERVAL '{days} days'
        """))
        this_period = result.scalar() or 0

        # Freshness distribution (this period)
        result = await conn.execute(text(f"""
            SELECT
                odds_freshness,
                COUNT(*) as count
            FROM odds_snapshots
            WHERE snapshot_type = 'lineup_confirmed'
              AND snapshot_at > NOW() - INTERVAL '{days} days'
            GROUP BY odds_freshness
        """))
        freshness_dist = {row.odds_freshness or 'unknown': row.count for row in result.fetchall()}

        # Delta to kickoff distribution (this period, in minutes)
        result = await conn.execute(text(f"""
            SELECT delta_to_kickoff_seconds
            FROM odds_snapshots
            WHERE snapshot_type = 'lineup_confirmed'
              AND snapshot_at > NOW() - INTERVAL '{days} days'
              AND delta_to_kickoff_seconds IS NOT NULL
        """))
        deltas = [row.delta_to_kickoff_seconds / 60 for row in result.fetchall()]  # Convert to minutes

        if deltas:
            delta_p50 = np.percentile(deltas, 50)
            delta_p90 = np.percentile(deltas, 90)
            delta_p10 = np.percentile(deltas, 10)
            delta_min = min(deltas)
            delta_max = max(deltas)
            # TARGET WINDOW: 45-75 minutes before kickoff
            in_target_window = sum(1 for d in deltas if 45 <= d <= 75)
            target_window_pct = round(in_target_window / len(deltas) * 100, 1) if deltas else 0
            # Build histogram buckets
            histogram = {
                "0-15min": sum(1 for d in deltas if 0 <= d < 15),
                "15-30min": sum(1 for d in deltas if 15 <= d < 30),
                "30-45min": sum(1 for d in deltas if 30 <= d < 45),
                "45-60min": sum(1 for d in deltas if 45 <= d < 60),
                "60-75min": sum(1 for d in deltas if 60 <= d < 75),
                "75-90min": sum(1 for d in deltas if 75 <= d < 90),
                "90+min": sum(1 for d in deltas if d >= 90),
            }
        else:
            delta_p50 = delta_p90 = delta_p10 = delta_min = delta_max = None
            target_window_pct = 0
            histogram = {}

        # Coverage by league (this period)
        result = await conn.execute(text(f"""
            SELECT
                m.league_id,
                COUNT(*) as count
            FROM odds_snapshots os
            JOIN matches m ON os.match_id = m.id
            WHERE os.snapshot_type = 'lineup_confirmed'
              AND os.snapshot_at > NOW() - INTERVAL '{days} days'
            GROUP BY m.league_id
            ORDER BY count DESC
        """))
        by_league = {row.league_id: row.count for row in result.fetchall()}

        # Coverage by bookmaker (this period)
        result = await conn.execute(text(f"""
            SELECT
                bookmaker,
                COUNT(*) as count
            FROM odds_snapshots
            WHERE snapshot_type = 'lineup_confirmed'
              AND snapshot_at > NOW() - INTERVAL '{days} days'
            GROUP BY bookmaker
            ORDER BY count DESC
        """))
        by_bookmaker = {row.bookmaker or 'unknown': row.count for row in result.fetchall()}

        # Weekly breakdown (last 8 weeks)
        result = await conn.execute(text("""
            SELECT
                DATE_TRUNC('week', snapshot_at) as week,
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE odds_freshness = 'live') as live_count
            FROM odds_snapshots
            WHERE snapshot_type = 'lineup_confirmed'
            GROUP BY DATE_TRUNC('week', snapshot_at)
            ORDER BY week DESC
            LIMIT 8
        """))
        weekly_breakdown = [
            {
                "week": row.week.strftime('%Y-%m-%d') if row.week else 'unknown',
                "total": row.total,
                "live": row.live_count,
                "live_pct": round(row.live_count / row.total * 100, 1) if row.total > 0 else 0
            }
            for row in result.fetchall()
        ]

        return {
            "total_real": total_real,
            "total_live_valid": total_live_valid,
            "this_period": this_period,
            "freshness_distribution": freshness_dist,
            "delta_to_kickoff_minutes": {
                "p10": round(delta_p10, 1) if delta_p10 else None,
                "p50": round(delta_p50, 1) if delta_p50 else None,
                "p90": round(delta_p90, 1) if delta_p90 else None,
                "min": round(delta_min, 1) if delta_min else None,
                "max": round(delta_max, 1) if delta_max else None,
                "target_window_pct": target_window_pct,  # % in 45-75 min window
                "histogram": histogram,
            },
            "by_league": by_league,
            "by_bookmaker": by_bookmaker,
            "weekly_breakdown": weekly_breakdown,
        }


async def check_pit_trigger(total_live: int) -> dict:
    """Check if PIT evaluation should be triggered."""
    triggers = {
        "preliminary_reached": total_live >= MILESTONE_PRELIMINARY,
        "full_reached": total_live >= MILESTONE_FULL,
        "current_n": total_live,
        "milestone_preliminary": MILESTONE_PRELIMINARY,
        "milestone_full": MILESTONE_FULL,
        "progress_preliminary_pct": min(100, round(total_live / MILESTONE_PRELIMINARY * 100, 1)),
        "progress_full_pct": min(100, round(total_live / MILESTONE_FULL * 100, 1)),
    }

    if total_live >= MILESTONE_FULL:
        triggers["action"] = "RUN_PIT_FULL"
        triggers["action_description"] = "Ejecutar evaluación PIT COMPLETA con --live-only"
    elif total_live >= MILESTONE_PRELIMINARY:
        triggers["action"] = "RUN_PIT_PRELIMINARY"
        triggers["action_description"] = "Ejecutar evaluación PIT PRELIMINAR con --live-only"
    else:
        triggers["action"] = "WAIT"
        triggers["action_description"] = f"Continuar acumulando datos. Faltan {MILESTONE_PRELIMINARY - total_live} para hito preliminar."

    return triggers


# =============================================================================
# REPORT GENERATION
# =============================================================================

async def generate_weekly_report(engine) -> dict:
    """Generate comprehensive weekly report."""
    report_time = datetime.utcnow()

    # Collect stats
    stats = await get_weekly_stats(engine, weeks_back=1)
    trigger_status = await check_pit_trigger(stats["total_live_valid"])

    # Calculate freshness percentages
    total_freshness = sum(stats["freshness_distribution"].values())
    freshness_pct = {}
    if total_freshness > 0:
        for k, v in stats["freshness_distribution"].items():
            freshness_pct[k] = round(v / total_freshness * 100, 1)

    report = {
        "report_type": "weekly_lineup_confirmed_audit",
        "generated_at": report_time.isoformat(),
        "period": "last_7_days",

        # Summary
        "summary": {
            "snapshots_this_week": stats["this_period"],
            "total_accumulated": stats["total_real"],
            "total_live_valid_for_pit": stats["total_live_valid"],
        },

        # Freshness analysis
        "freshness": {
            "distribution_counts": stats["freshness_distribution"],
            "distribution_pct": freshness_pct,
            "live_pct": freshness_pct.get("live", 0),
        },

        # Timing analysis
        "delta_to_kickoff": stats["delta_to_kickoff_minutes"],

        # Coverage
        "coverage": {
            "by_league": stats["by_league"],
            "by_bookmaker": stats["by_bookmaker"],
        },

        # Historical trend
        "weekly_trend": stats["weekly_breakdown"],

        # PIT trigger status
        "pit_evaluation": {
            "trigger_status": trigger_status,
            "frozen_config": FROZEN_CONFIG,
            "note": "Config is PRE-REGISTERED and FROZEN. Do NOT modify.",
        },
    }

    return report


def print_report(report: dict):
    """Print report to console in readable format."""
    print("=" * 70)
    print("REPORTE SEMANAL - AUDITORÍA lineup_confirmed PIT")
    print("=" * 70)
    print(f"Generado: {report['generated_at']}")
    print()

    # Summary
    print("📊 RESUMEN")
    print("-" * 50)
    s = report["summary"]
    print(f"  Snapshots esta semana:      {s['snapshots_this_week']}")
    print(f"  Total acumulado:            {s['total_accumulated']}")
    print(f"  Total live válido (PIT):    {s['total_live_valid_for_pit']}")
    print()

    # Freshness
    print("🔄 FRESHNESS DISTRIBUTION")
    print("-" * 50)
    f = report["freshness"]
    for key, pct in f["distribution_pct"].items():
        count = f["distribution_counts"].get(key, 0)
        marker = "✅" if key == "live" else ""
        print(f"  {key:<12} {count:>6} ({pct:>5.1f}%) {marker}")
    print(f"\n  → % Live: {f['live_pct']:.1f}%")
    print()

    # Delta to kickoff
    print("⏱️  DELTA TO KICKOFF (minutos antes del partido)")
    print("-" * 50)
    d = report["delta_to_kickoff"]
    if d["p50"]:
        print(f"  p10:           {d['p10']} min")
        print(f"  p50 (mediana): {d['p50']} min")
        print(f"  p90:           {d['p90']} min")
        print(f"  Rango:         {d['min']} - {d['max']} min")
        print()
        target_pct = d.get('target_window_pct', 0)
        target_marker = "✅" if target_pct >= 50 else ("🔶" if target_pct >= 25 else "❌")
        print(f"  % en ventana objetivo (45-75 min): {target_pct}% {target_marker}")
        print()
        print("  Histograma:")
        histogram = d.get('histogram', {})
        for bucket, count in histogram.items():
            bar = "█" * min(count, 20)
            target = " ← TARGET" if bucket in ["45-60min", "60-75min"] else ""
            print(f"    {bucket:<10} {count:>3} {bar}{target}")
    else:
        print("  Sin datos de timing")
    print()

    # Coverage by league
    print("🏆 COBERTURA POR LIGA (top 10)")
    print("-" * 50)
    leagues = report["coverage"]["by_league"]
    for i, (league_id, count) in enumerate(sorted(leagues.items(), key=lambda x: -x[1])[:10]):
        print(f"  Liga {league_id:<4} {count:>4} snapshots")
    print()

    # Coverage by bookmaker
    print("📖 COBERTURA POR BOOKMAKER")
    print("-" * 50)
    bookmakers = report["coverage"]["by_bookmaker"]
    for bm, count in sorted(bookmakers.items(), key=lambda x: -x[1]):
        print(f"  {bm:<20} {count:>4}")
    print()

    # Weekly trend
    print("📈 TENDENCIA SEMANAL (últimas 8 semanas)")
    print("-" * 50)
    print(f"  {'Semana':<12} {'Total':>6} {'Live':>6} {'Live%':>7}")
    for week in report["weekly_trend"]:
        print(f"  {week['week']:<12} {week['total']:>6} {week['live']:>6} {week['live_pct']:>6.1f}%")
    print()

    # PIT Trigger
    print("🎯 STATUS EVALUACIÓN PIT")
    print("-" * 50)
    pit = report["pit_evaluation"]["trigger_status"]
    print(f"  N actual (live válido): {pit['current_n']}")
    print(f"  Hito preliminar (N≥{pit['milestone_preliminary']}): {pit['progress_preliminary_pct']}%")
    print(f"  Hito completo (N≥{pit['milestone_full']}): {pit['progress_full_pct']}%")
    print()
    print(f"  → ACCIÓN: {pit['action']}")
    print(f"    {pit['action_description']}")
    print()

    # Frozen config reminder
    print("🔒 CONFIG PRINCIPAL (CONGELADA)")
    print("-" * 50)
    cfg = report["pit_evaluation"]["frozen_config"]
    print(f"  Threshold:  {cfg['edge_threshold']}")
    print(f"  Stake:      {cfg['stake']} (fijo)")
    print(f"  Gate:       {'Activo' if cfg['use_gate'] else 'Inactivo'}")
    print(f"  Odds:       {cfg['odds_type']}")
    print("  ⚠️  NO MODIFICAR - Pre-registrado para evitar p-hacking")
    print()

    print("=" * 70)


async def main():
    engine = create_async_engine(DATABASE_URL)

    # Generate report
    report = await generate_weekly_report(engine)

    # Print to console
    print_report(report)

    # Save to JSON
    output_file = f"scripts/lineup_audit_{datetime.utcnow().strftime('%Y%m%d')}.json"
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Reporte guardado en: {output_file}")

    # Check if PIT evaluation should be triggered
    pit_status = report["pit_evaluation"]["trigger_status"]
    if pit_status["action"] in ["RUN_PIT_PRELIMINARY", "RUN_PIT_FULL"]:
        print()
        print("🚨 TRIGGER ACTIVADO")
        print("Ejecutar evaluación PIT con:")
        print("  DATABASE_URL=... python scripts/evaluate_pit_live_only.py")

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
