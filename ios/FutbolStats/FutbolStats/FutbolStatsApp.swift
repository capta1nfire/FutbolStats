import SwiftUI

// MARK: - App Color Palette

/// Consistent color palette derived from the main gradient background
enum AppColors {
    // Background gradient colors
    static let backgroundDark = Color(red: 0.02, green: 0.02, blue: 0.06)
    static let backgroundLight = Color(red: 0.034, green: 0.034, blue: 0.10)

    // Surface colors (for cards, circles, etc.)
    static let surface = Color(white: 0.08)
    static let surfaceElevated = Color(white: 0.12)
    static let surfaceHighlight = Color(white: 0.15)

    // Text colors
    static let textPrimary = Color.white
    static let textSecondary = Color.white.opacity(0.7)
    static let textTertiary = Color.white.opacity(0.5)
    static let textMuted = Color.white.opacity(0.35)

    // Accent colors
    static let accent = Color.cyan
    static let accentSecondary = Color.blue

    // Main background gradient
    static var backgroundGradient: LinearGradient {
        LinearGradient(
            stops: [
                .init(color: backgroundDark, location: 0),
                .init(color: backgroundDark, location: 0.7),
                .init(color: backgroundLight, location: 1.0)
            ],
            startPoint: .top,
            endPoint: .bottom
        )
    }
}

@main
struct FutbolStatsApp: App {
    @Environment(\.scenePhase) private var scenePhase

    init() {
        // Set tab bar appearance - white for selected items
        UITabBar.appearance().tintColor = .white
        UITabBar.appearance().unselectedItemTintColor = .gray
    }

    var body: some Scene {
        WindowGroup {
            MainTabView()
        }
        .onChange(of: scenePhase) { _, newPhase in
            switch newPhase {
            case .active:
                // Resume live score polling when app becomes active
                LiveScoreManager.shared.onAppBecameActive()
            case .inactive, .background:
                // Stop polling and log telemetry when app goes to background
                LiveScoreManager.shared.onAppBecameInactive()
                ImageCache.shared.logStats()
            @unknown default:
                break
            }
        }
    }
}

// MARK: - Tab Enum

enum AppTab: Int, Hashable {
    case home = 0
    case teams = 1
    case match = 2
    case competitions = 3
    case search = 4
}

// MARK: - Main Tab View

struct MainTabView: View {
    @State private var showingHomeMenu = false
    @State private var navigateToDashboard = false
    @State private var selectedTab: AppTab = .match  // Default to Match tab

    var body: some View {
        TabView(selection: $selectedTab) {
            // Home - shows menu
            HomeMenuView(showingMenu: $showingHomeMenu, navigateToDashboard: $navigateToDashboard)
                .tabItem {
                    Label("Home", systemImage: "house.fill")
                }
                .tag(AppTab.home)

            TeamsView()
                .tabItem {
                    Label("Teams", systemImage: "person.3.fill")
                }
                .tag(AppTab.teams)

            PredictionsListView()
                .tabItem {
                    Label("Match", systemImage: "sportscourt.fill")
                }
                .tag(AppTab.match)

            CompetitionsView()
                .tabItem {
                    Label("Competitions", systemImage: "trophy.fill")
                }
                .tag(AppTab.competitions)

            // Search tab - independent search view
            SearchView()
                .tabItem {
                    Label("Search", systemImage: "magnifyingglass")
                }
                .tag(AppTab.search)
        }
        .tint(.white)
    }
}

// MARK: - Home Menu View

struct HomeMenuView: View {
    @Binding var showingMenu: Bool
    @Binding var navigateToDashboard: Bool

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                VStack(spacing: 20) {
                    // User avatar
                    Image("UserAvatar")
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: 80, height: 80)
                        .clipShape(Circle())
                        .padding(.top, 40)

                    Text("FutbolStats")
                        .font(.title2)
                        .fontWeight(.bold)
                        .foregroundStyle(.white)

                    // Menu options
                    VStack(spacing: 0) {
                        NavigationLink(destination: DashboardView()) {
                            menuRow(icon: "gauge.with.dots.needle.33percent", title: "Dashboard")
                        }

                        Divider().background(Color.gray.opacity(0.3))

                        NavigationLink(destination: Text("My Teams").foregroundStyle(.white)) {
                            menuRow(icon: "heart.fill", title: "My Teams")
                        }

                        Divider().background(Color.gray.opacity(0.3))

                        NavigationLink(destination: Text("Settings").foregroundStyle(.white)) {
                            menuRow(icon: "gear", title: "Settings")
                        }
                    }
                    .background(Color(white: 0.1))
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal, 20)

                    Spacer()
                }
            }
            .navigationBarHidden(true)
        }
    }

    private func menuRow(icon: String, title: String) -> some View {
        HStack(spacing: 16) {
            Image(systemName: icon)
                .font(.system(size: 20))
                .foregroundStyle(.gray)
                .frame(width: 28)

            Text(title)
                .font(.body)
                .foregroundStyle(.white)

            Spacer()

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundStyle(.gray)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }
}

