import SwiftUI

/// Match statistics table view - displays stats like halftime broadcast
/// Shows possession, shots, xG, etc. in a compact, scannable format
struct MatchStatsTableView: View {
    let stats: MatchStats
    let homeTeam: String
    let awayTeam: String

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Header
            HStack {
                Image(systemName: "chart.bar.fill")
                    .foregroundStyle(.cyan)
                Text("Estadísticas")
                    .font(.headline)
                    .fontWeight(.bold)
                    .foregroundStyle(.white)
            }

            // Stats rows
            VStack(spacing: 12) {
                // Possession (special: bar visualization)
                if let homePoss = stats.home?.ballPossession,
                   let awayPoss = stats.away?.ballPossession {
                    possessionRow(home: homePoss, away: awayPoss)
                }

                // xG (Expected Goals)
                if let homeXG = stats.home?.expectedGoals,
                   let awayXG = stats.away?.expectedGoals {
                    statRow(
                        label: "xG",
                        homeValue: homeXG,
                        awayValue: awayXG,
                        icon: "target"
                    )
                }

                // Shots on Target
                if let homeSoT = stats.home?.shotsOnGoal,
                   let awaySoT = stats.away?.shotsOnGoal {
                    statRow(
                        label: "Tiros a puerta",
                        homeValue: "\(homeSoT)",
                        awayValue: "\(awaySoT)",
                        icon: "scope"
                    )
                }

                // Total Shots
                if let homeShots = stats.home?.totalShots,
                   let awayShots = stats.away?.totalShots {
                    statRow(
                        label: "Tiros totales",
                        homeValue: "\(homeShots)",
                        awayValue: "\(awayShots)",
                        icon: "circle.circle"
                    )
                }

                // Corners
                if let homeCorners = stats.home?.cornerKicks,
                   let awayCorners = stats.away?.cornerKicks {
                    statRow(
                        label: "Corners",
                        homeValue: "\(homeCorners)",
                        awayValue: "\(awayCorners)",
                        icon: "flag.fill"
                    )
                }

                // Fouls
                if let homeFouls = stats.home?.fouls,
                   let awayFouls = stats.away?.fouls {
                    statRow(
                        label: "Faltas",
                        homeValue: "\(homeFouls)",
                        awayValue: "\(awayFouls)",
                        icon: "exclamationmark.triangle"
                    )
                }

                // Yellow Cards
                if let homeYellow = stats.home?.yellowCards,
                   let awayYellow = stats.away?.yellowCards,
                   (homeYellow > 0 || awayYellow > 0) {
                    statRow(
                        label: "Amarillas",
                        homeValue: "\(homeYellow)",
                        awayValue: "\(awayYellow)",
                        icon: "rectangle.fill",
                        iconColor: .yellow
                    )
                }

                // Red Cards
                if let homeRed = stats.home?.redCards,
                   let awayRed = stats.away?.redCards,
                   (homeRed > 0 || awayRed > 0) {
                    statRow(
                        label: "Rojas",
                        homeValue: "\(homeRed)",
                        awayValue: "\(awayRed)",
                        icon: "rectangle.fill",
                        iconColor: .red
                    )
                }

                // Goalkeeper Saves
                if let homeSaves = stats.home?.goalkeeperSaves,
                   let awaySaves = stats.away?.goalkeeperSaves {
                    statRow(
                        label: "Paradas",
                        homeValue: "\(homeSaves)",
                        awayValue: "\(awaySaves)",
                        icon: "hand.raised.fill"
                    )
                }
            }
        }
        .padding(16)
        .background(
            RoundedRectangle(cornerRadius: 16)
                .fill(Color(white: 0.08))
        )
    }

    // MARK: - Possession Row (with bar)

    private func possessionRow(home: String, away: String) -> some View {
        let homePercent = parsePercent(home)
        let awayPercent = parsePercent(away)

        return VStack(spacing: 8) {
            HStack {
                Text(home)
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .foregroundStyle(.blue)

                Spacer()

                HStack(spacing: 4) {
                    Image(systemName: "circle.hexagongrid.fill")
                        .font(.caption)
                        .foregroundStyle(.gray)
                    Text("Posesión")
                        .font(.caption)
                        .foregroundStyle(.gray)
                }

                Spacer()

                Text(away)
                    .font(.subheadline)
                    .fontWeight(.bold)
                    .foregroundStyle(.red)
            }

            // Possession bar
            GeometryReader { geo in
                HStack(spacing: 0) {
                    Rectangle()
                        .fill(Color.blue)
                        .frame(width: geo.size.width * homePercent)

                    Rectangle()
                        .fill(Color.red)
                        .frame(width: geo.size.width * awayPercent)
                }
                .frame(height: 8)
                .clipShape(RoundedRectangle(cornerRadius: 4))
            }
            .frame(height: 8)
        }
    }

    // MARK: - Regular Stat Row

    private func statRow(
        label: String,
        homeValue: String,
        awayValue: String,
        icon: String,
        iconColor: Color = .gray
    ) -> some View {
        HStack {
            Text(homeValue)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(.white)
                .frame(width: 40, alignment: .leading)

            Spacer()

            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption)
                    .foregroundStyle(iconColor)
                Text(label)
                    .font(.caption)
                    .foregroundStyle(.gray)
            }

            Spacer()

            Text(awayValue)
                .font(.subheadline)
                .fontWeight(.semibold)
                .foregroundStyle(.white)
                .frame(width: 40, alignment: .trailing)
        }
    }

    // MARK: - Helpers

    private func parsePercent(_ value: String) -> Double {
        // "50%" -> 0.50
        let cleaned = value.replacingOccurrences(of: "%", with: "")
        if let percent = Double(cleaned) {
            return percent / 100.0
        }
        return 0.5
    }
}

// MARK: - Preview

#Preview {
    ScrollView {
        MatchStatsTableView(
            stats: MatchStats(
                home: TeamStats(
                    ballPossession: "54%",
                    totalShots: 15,
                    shotsOnGoal: 6,
                    shotsOffGoal: 5,
                    blockedShots: 4,
                    shotsInsidebox: 10,
                    shotsOutsidebox: 5,
                    fouls: 12,
                    cornerKicks: 7,
                    offsides: 3,
                    yellowCards: 2,
                    redCards: 0,
                    goalkeeperSaves: 3,
                    totalPasses: 450,
                    passesAccurate: 380,
                    passesPct: "84%",
                    expectedGoals: "1.85"
                ),
                away: TeamStats(
                    ballPossession: "46%",
                    totalShots: 8,
                    shotsOnGoal: 3,
                    shotsOffGoal: 3,
                    blockedShots: 2,
                    shotsInsidebox: 5,
                    shotsOutsidebox: 3,
                    fouls: 15,
                    cornerKicks: 3,
                    offsides: 1,
                    yellowCards: 3,
                    redCards: 1,
                    goalkeeperSaves: 4,
                    totalPasses: 320,
                    passesAccurate: 250,
                    passesPct: "78%",
                    expectedGoals: "0.72"
                )
            ),
            homeTeam: "Inter",
            awayTeam: "Napoli"
        )
        .padding()
    }
    .background(Color.black)
    .preferredColorScheme(.dark)
}
