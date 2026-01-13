import Foundation

/// Performance logger for iOS instrumentation.
/// Sends timing data to /debug/log endpoint for analysis.
///
/// Activation:
/// - DEBUG builds: always enabled
/// - RELEASE builds: enabled only if UserDefaults "DEBUG_PERF_LOGS" == "true"
///
/// Transport:
/// - Fire-and-forget POST (no retries, 1s timeout)
/// - Uses existing dashboard token for auth
actor PerfLogger {
    static let shared = PerfLogger()

    private let endpoint = "https://web-production-f2de9.up.railway.app/debug/log"
    private let session: URLSession

    /// Check if perf logging is enabled
    private var isEnabled: Bool {
        #if DEBUG
        return true
        #else
        return UserDefaults.standard.string(forKey: "DEBUG_PERF_LOGS") == "true"
        #endif
    }

    private init() {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 1.0  // 1s timeout
        config.timeoutIntervalForResource = 1.0
        self.session = URLSession(configuration: config)
    }

    /// Log a performance event (fire-and-forget)
    /// - Parameters:
    ///   - endpoint: The operation being measured (e.g., "loadPredictions")
    ///   - message: Event type (e.g., "start", "network_done", "end")
    ///   - data: Timing data dictionary
    ///   - hypothesisId: Optional hypothesis identifier for correlation
    nonisolated func log(
        endpoint: String,
        message: String,
        data: [String: Any],
        hypothesisId: String = "PERF"
    ) {
        // Fire-and-forget: don't block caller
        Task.detached(priority: .utility) {
            await self._log(endpoint: endpoint, message: message, data: data, hypothesisId: hypothesisId)
        }
    }

    private func _log(
        endpoint: String,
        message: String,
        data: [String: Any],
        hypothesisId: String
    ) async {
        guard isEnabled else { return }

        // Print locally for Xcode console
        let dataStr = data.map { "\($0.key)=\($0.value)" }.joined(separator: ", ")
        print("[PerfLog] \(endpoint) | \(message) | \(dataStr)")

        let payload: [String: Any] = [
            "component": "[IOS]",
            "endpoint": endpoint,
            "message": message,
            "data": data,
            "hypothesisId": hypothesisId,
            "timestamp": Int(Date().timeIntervalSince1970 * 1000)
        ]

        guard let url = URL(string: self.endpoint),
              let jsonData = try? JSONSerialization.data(withJSONObject: payload) else {
            return
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        // Use dashboard token for auth
        if let token = AppConfiguration.dashboardToken {
            request.setValue(token, forHTTPHeaderField: "X-Dashboard-Token")
        }

        request.httpBody = jsonData

        // Fire-and-forget: ignore result
        _ = try? await session.data(for: request)
    }
}

// MARK: - Timing Helper

/// Helper to measure execution time
struct PerfTimer {
    let start: Date

    init() {
        self.start = Date()
    }

    /// Elapsed time in milliseconds
    var elapsedMs: Double {
        Date().timeIntervalSince(start) * 1000
    }
}
