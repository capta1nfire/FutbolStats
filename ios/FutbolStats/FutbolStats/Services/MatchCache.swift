import Foundation

/// Cached live match data for overlay updates.
/// Only stores fields that change during live matches.
struct CachedMatchData {
    let status: String?
    let elapsed: Int?
    let elapsedExtra: Int?  // Added/injury time (e.g., 3 for 90+3)
    let homeGoals: Int?
    let awayGoals: Int?
    let updatedAt: Date

    /// TTL based on match state (30s for live, 5min for finished)
    var ttlSeconds: TimeInterval {
        guard let status = status else { return 300 }
        let liveStatuses = ["1H", "2H", "HT", "ET", "BT", "P", "LIVE"]
        return liveStatuses.contains(status) ? 30 : 300
    }

    /// Check if cache entry is still fresh
    var isFresh: Bool {
        Date().timeIntervalSince(updatedAt) < ttlSeconds
    }
}

/// Singleton cache for live match updates.
/// MatchDetailView writes here, PredictionsListView reads as overlay.
///
/// Guardrails (Auditor-approved):
/// 1. Cache is overlay, not source of truth - never replaces API data structure
/// 2. TTL by state: 30s for live matches, 5min for finished
/// 3. Does not modify list structure - only overlays specific fields
/// 4. Minimal data to avoid heavy re-renders
@MainActor
final class MatchCache {
    static let shared = MatchCache()

    /// In-memory cache: matchId -> CachedMatchData
    private var cache: [Int: CachedMatchData] = [:]

    /// Notification name for cache updates (allows Views to subscribe)
    static let didUpdateNotification = Notification.Name("MatchCacheDidUpdate")

    private init() {}

    // MARK: - Write (from MatchDetailView)

    /// Update cache with fresh data from MatchDetailView polling.
    /// - Parameters:
    ///   - matchId: The match ID
    ///   - status: Current match status (1H, HT, 2H, FT, etc.)
    ///   - elapsed: Current minute (nil for non-live)
    ///   - elapsedExtra: Added/injury time minutes (e.g., 3 for 90+3)
    ///   - homeGoals: Home team goals
    ///   - awayGoals: Away team goals
    func update(
        matchId: Int,
        status: String?,
        elapsed: Int?,
        elapsedExtra: Int?,
        homeGoals: Int?,
        awayGoals: Int?
    ) {
        let entry = CachedMatchData(
            status: status,
            elapsed: elapsed,
            elapsedExtra: elapsedExtra,
            homeGoals: homeGoals,
            awayGoals: awayGoals,
            updatedAt: Date()
        )
        cache[matchId] = entry

        // Post notification for interested Views
        NotificationCenter.default.post(
            name: Self.didUpdateNotification,
            object: matchId
        )

        // Format elapsed display for logging
        let elapsedStr = elapsed.map { extra in
            elapsedExtra.map { "\(extra)+\($0)" } ?? "\(extra)"
        } ?? "-1"
        print("[MatchCache] Updated match \(matchId): status=\(status ?? "nil"), elapsed=\(elapsedStr), score=\(homeGoals ?? 0)-\(awayGoals ?? 0)")
    }

    /// Convenience method to update from a MatchPrediction
    func update(from prediction: MatchPrediction) {
        guard let matchId = prediction.matchId else { return }
        update(
            matchId: matchId,
            status: prediction.status,
            elapsed: prediction.elapsed,
            elapsedExtra: prediction.elapsedExtra,
            homeGoals: prediction.homeGoals,
            awayGoals: prediction.awayGoals
        )
    }

    // MARK: - Read (from PredictionsListView)

    /// Get cached data for a match if fresh.
    /// Returns nil if not cached or stale (TTL expired).
    func get(matchId: Int) -> CachedMatchData? {
        guard let entry = cache[matchId], entry.isFresh else {
            return nil
        }
        return entry
    }

    /// Check if we have fresh data for a match
    func hasFreshData(for matchId: Int) -> Bool {
        get(matchId: matchId) != nil
    }

    // MARK: - Overlay Application

    /// Apply cached overlay to a prediction (returns new values, doesn't mutate).
    /// This is the core "overlay" pattern - only update live fields if cache is fresher.
    /// - Parameters:
    ///   - prediction: Original prediction from API
    /// - Returns: Tuple of overlayed values (status, elapsed, elapsedExtra, homeGoals, awayGoals) or nil if no fresh cache
    func overlay(for prediction: MatchPrediction) -> (status: String?, elapsed: Int?, elapsedExtra: Int?, homeGoals: Int?, awayGoals: Int?)? {
        guard let matchId = prediction.matchId,
              let cached = get(matchId: matchId) else {
            return nil
        }

        return (cached.status, cached.elapsed, cached.elapsedExtra, cached.homeGoals, cached.awayGoals)
    }

    // MARK: - Maintenance

    /// Remove stale entries (call periodically if needed)
    func pruneStale() {
        let before = cache.count
        cache = cache.filter { $0.value.isFresh }
        let pruned = before - cache.count
        if pruned > 0 {
            print("[MatchCache] Pruned \(pruned) stale entries")
        }
    }

    /// Clear all cache (e.g., on app background)
    func clear() {
        cache.removeAll()
        print("[MatchCache] Cleared")
    }

    /// Current cache size (for debugging)
    var count: Int { cache.count }
}
