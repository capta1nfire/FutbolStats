import SwiftUI

@main
struct FutbolStatsApp: App {
    var body: some Scene {
        WindowGroup {
            MainTabView()
        }
    }
}

// MARK: - Main Tab View

struct MainTabView: View {
    var body: some View {
        TabView {
            PredictionsListView()
                .tabItem {
                    Label("Predictions", systemImage: "chart.bar.fill")
                }

            TeamsView()
                .tabItem {
                    Label("Teams", systemImage: "person.3.fill")
                }

            CompetitionsView()
                .tabItem {
                    Label("Competitions", systemImage: "trophy.fill")
                }
        }
    }
}
