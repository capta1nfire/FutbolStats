import Foundation

// MARK: - OPS Dashboard JSON (/dashboard/ops.json)

/// Wrapper for /dashboard/ops.json response (backend returns { data: {...}, cache_age_seconds: ... })
struct OpsDashboardWrapper: Decodable {
    let data: OpsDashboardResponse
    let cacheAgeSeconds: Double?

    enum CodingKeys: String, CodingKey {
        case data
        case cacheAgeSeconds = "cache_age_seconds"
    }
}

struct OpsDashboardResponse: Decodable {
    let generatedAt: String
    let leagueMode: String?
    let trackedLeaguesCount: Int?
    let lastSyncAt: String?
    let budget: BudgetStatus?
    let pit: OpsPIT?
    let movement: OpsMovement?
    let statsBackfill: OpsStatsBackfill?
    let upcoming: OpsUpcoming?
    let progress: OpsProgress?

    enum CodingKeys: String, CodingKey {
        case generatedAt = "generated_at"
        case leagueMode = "league_mode"
        case trackedLeaguesCount = "tracked_leagues_count"
        case lastSyncAt = "last_sync_at"
        case budget
        case pit
        case movement
        case statsBackfill = "stats_backfill"
        case upcoming
        case progress
    }
}

struct BudgetStatus: Decodable {
    let status: String?
    let plan: String?
    let requestsToday: Int?
    let requestsLimit: Int?
    let tokensResetTz: String?
    let tokensResetLocalTime: String?
    let tokensResetAtLA: String?
    let tokensResetAtUTC: String?
    let tokensResetNote: String?
    let error: String?

    enum CodingKeys: String, CodingKey {
        case status
        case plan
        case requestsToday = "requests_today"
        case requestsLimit = "requests_limit"
        case tokensResetTz = "tokens_reset_tz"
        case tokensResetLocalTime = "tokens_reset_local_time"
        case tokensResetAtLA = "tokens_reset_at_la"
        case tokensResetAtUTC = "tokens_reset_at_utc"
        case tokensResetNote = "tokens_reset_note"
        case error
    }
}

struct OpsPIT: Decodable {
    let live60m: Int?
    let live24h: Int?
    let deltaToKickoff60m: [OpsDeltaToKickoffBin]?
    let latest: [OpsLatestPIT]?

    enum CodingKeys: String, CodingKey {
        case live60m = "live_60m"
        case live24h = "live_24h"
        case deltaToKickoff60m = "delta_to_kickoff_60m"
        case latest
    }
}

struct OpsDeltaToKickoffBin: Decodable, Identifiable {
    let minToKo: Int
    let count: Int
    var id: String { "\(minToKo)" }

    enum CodingKeys: String, CodingKey {
        case minToKo = "min_to_ko"
        case count
    }
}

struct OpsLatestPIT: Decodable, Identifiable {
    let snapshotAt: String?
    let matchId: Int?
    let leagueId: Int?
    let leagueName: String?
    let oddsFreshness: String?
    let deltaToKickoffMinutes: Double?
    let odds: OpsOdds?
    let bookmaker: String?

    var id: String { "\(matchId ?? 0)-\(snapshotAt ?? UUID().uuidString)" }

    enum CodingKeys: String, CodingKey {
        case snapshotAt = "snapshot_at"
        case matchId = "match_id"
        case leagueId = "league_id"
        case leagueName = "league_name"
        case oddsFreshness = "odds_freshness"
        case deltaToKickoffMinutes = "delta_to_kickoff_minutes"
        case odds
        case bookmaker
    }
}

struct OpsOdds: Decodable {
    let home: Double?
    let draw: Double?
    let away: Double?
}

struct OpsMovement: Decodable {
    let lineupMovement24h: Int?
    let marketMovement24h: Int?

    enum CodingKeys: String, CodingKey {
        case lineupMovement24h = "lineup_movement_24h"
        case marketMovement24h = "market_movement_24h"
    }
}

struct OpsStatsBackfill: Decodable {
    let finished72hWithStats: Int?
    let finished72hMissingStats: Int?

    enum CodingKeys: String, CodingKey {
        case finished72hWithStats = "finished_72h_with_stats"
        case finished72hMissingStats = "finished_72h_missing_stats"
    }
}

struct OpsUpcoming: Decodable {
    let byLeague24h: [OpsUpcomingLeague]?

    enum CodingKeys: String, CodingKey {
        case byLeague24h = "by_league_24h"
    }
}

struct OpsUpcomingLeague: Decodable, Identifiable {
    let leagueId: Int
    let leagueName: String?
    let upcoming24h: Int

    var id: Int { leagueId }

    enum CodingKeys: String, CodingKey {
        case leagueId = "league_id"
        case leagueName = "league_name"
        case upcoming24h = "upcoming_24h"
    }
}

struct OpsProgress: Decodable {
    let pitSnapshots30d: Int?
    let targetPitSnapshots30d: Int?
    let pitBets30d: Int?
    let targetPitBets30d: Int?
    let baselineCoveragePct: Double?
    let pitWithBaseline: Int?
    let pitTotalForBaseline: Int?
    let targetBaselineCoveragePct: Int?
    let readyForRetest: Bool?

    enum CodingKeys: String, CodingKey {
        case pitSnapshots30d = "pit_snapshots_30d"
        case targetPitSnapshots30d = "target_pit_snapshots_30d"
        case pitBets30d = "pit_bets_30d"
        case targetPitBets30d = "target_pit_bets_30d"
        case baselineCoveragePct = "baseline_coverage_pct"
        case pitWithBaseline = "pit_with_baseline"
        case pitTotalForBaseline = "pit_total_for_baseline"
        case targetBaselineCoveragePct = "target_baseline_coverage_pct"
        case readyForRetest = "ready_for_retest"
    }
}

// MARK: - Alpha Progress Snapshots (/dashboard/ops/progress_snapshots.json)

struct AlphaProgressSnapshotsResponse: Decodable {
    let count: Int
    let limit: Int
    let snapshots: [AlphaProgressSnapshotItem]
}

struct AlphaProgressSnapshotItem: Decodable, Identifiable {
    let id: Int
    let capturedAt: String?
    let payload: AlphaProgressSnapshotPayload?
    let source: String?
    let appCommit: String?

    enum CodingKeys: String, CodingKey {
        case id
        case capturedAt = "captured_at"
        case payload
        case source
        case appCommit = "app_commit"
    }
}

struct AlphaProgressSnapshotPayload: Decodable {
    let generatedAt: String?
    let leagueMode: String?
    let trackedLeaguesCount: Int?
    let progress: OpsProgress?

    enum CodingKeys: String, CodingKey {
        case generatedAt = "generated_at"
        case leagueMode = "league_mode"
        case trackedLeaguesCount = "tracked_leagues_count"
        case progress
    }
}

// MARK: - PIT Dashboard JSON (/dashboard/pit.json)

struct PITDashboardResponse: Decodable {
    let source: String?
    let daily: PITDailyLiveOnlyReport?
    let weekly: PITWeeklyConsolidatedReport?
    let error: String?
    let cacheAgeSeconds: Double?

    enum CodingKeys: String, CodingKey {
        case source
        case daily
        case weekly
        case error
        case cacheAgeSeconds = "cache_age_seconds"
    }
}

struct PITDailyLiveOnlyReport: Decodable {
    let generatedAt: String?
    let protocolVersion: String?
    let counts: PITDailyCounts?
    let brier: PITBrier?
    let betting: PITBetting?
    let phase: String?
    let interpretation: PITInterpretation?

    enum CodingKeys: String, CodingKey {
        case generatedAt = "generated_at"
        case protocolVersion = "protocol_version"
        case counts
        case brier
        case betting
        case phase
        case interpretation
    }
}

struct PITDailyCounts: Decodable {
    let nPitValid1090: Int?
    let nPitValidIdeal4575: Int?
    let nWithPitSafePredictions: Int?

    enum CodingKeys: String, CodingKey {
        case nPitValid1090 = "n_pit_valid_10_90"
        case nPitValidIdeal4575 = "n_pit_valid_ideal_45_75"
        case nWithPitSafePredictions = "n_with_pit_safe_predictions"
    }
}

struct PITBrier: Decodable {
    let nWithPredictions: Int?
    let brierModel: Double?
    let brierMarket: Double?
    let brierUniform: Double?
    let skillVsMarket: Double?
    let skillVsUniform: Double?

    enum CodingKeys: String, CodingKey {
        case nWithPredictions = "n_with_predictions"
        case brierModel = "brier_model"
        case brierMarket = "brier_market"
        case brierUniform = "brier_uniform"
        case skillVsMarket = "skill_vs_market"
        case skillVsUniform = "skill_vs_uniform"
    }
}

struct PITBetting: Decodable {
    let nBets: Int?
    let roi: Double?
    let roiCi95Low: Double?
    let roiCi95High: Double?
    let roiCiStatus: String?
    let ev: Double?
    let evCi95Low: Double?
    let evCi95High: Double?
    let evCiStatus: String?

    enum CodingKeys: String, CodingKey {
        case nBets = "n_bets"
        case roi
        case roiCi95Low = "roi_ci95_low"
        case roiCi95High = "roi_ci95_high"
        case roiCiStatus = "roi_ci_status"
        case ev
        case evCi95Low = "ev_ci95_low"
        case evCi95High = "ev_ci95_high"
        case evCiStatus = "ev_ci_status"
    }
}

struct PITInterpretation: Decodable {
    let confidence: String?
    let verdict: String?
    let bulletNotes: [String]?

    enum CodingKeys: String, CodingKey {
        case confidence
        case verdict
        case bulletNotes = "bullet_notes"
    }
}

struct PITWeeklyConsolidatedReport: Decodable {
    let generatedAt: String?
    let summary: PITWeeklySummary?
    let latestMetrics: PITWeeklyLatestMetrics?
    let recommendation: String?

    enum CodingKeys: String, CodingKey {
        case generatedAt = "generated_at"
        case summary
        case latestMetrics = "latest_metrics"
        case recommendation
    }
}

struct PITWeeklySummary: Decodable {
    let principalN: Int?
    let principalStatus: String?
    let idealN: Int?
    let idealStatus: String?
    let edgeDiagnostic: String?
    let qualityScore: Double?

    enum CodingKeys: String, CodingKey {
        case principalN = "principal_n"
        case principalStatus = "principal_status"
        case idealN = "ideal_n"
        case idealStatus = "ideal_status"
        case edgeDiagnostic = "edge_diagnostic"
        case qualityScore = "quality_score"
    }
}

struct PITWeeklyLatestMetrics: Decodable {
    let phase: String?
    let brier: PITBrier?
    let betting: PITBetting?
    let interpretation: PITInterpretation?
}


