import Foundation

/// Central app configuration (safe defaults, no secrets committed).
///
/// How to provide DASHBOARD token (preferred):
/// - Add `DASHBOARD_TOKEN` to the app Info.plist (per environment/config).
/// - Or set UserDefaults key `dashboard_token` for local/dev testing.
enum AppConfiguration {
    /// Dashboard token used by backend to authorize dashboard JSON endpoints.
    ///
    /// Backend supports:
    /// - Header: `X-Dashboard-Token: <token>` (preferred)
    /// - Query:  `?token=<token>` (avoid in production logs)
    static var dashboardToken: String? {
        // 1) Dev override (useful for local testing without changing plist)
        if let t = UserDefaults.standard.string(forKey: "dashboard_token"), !t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return t.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        // 2) Info.plist configuration
        if let t = Bundle.main.object(forInfoDictionaryKey: "DASHBOARD_TOKEN") as? String, !t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return t.trimmingCharacters(in: .whitespacesAndNewlines)
        }

        return nil
    }
}


