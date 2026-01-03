import SwiftUI

struct LeagueStandingsView: View {
    let leagueId: Int
    let homeTeamId: Int?
    let awayTeamId: Int?

    @State private var standings: [StandingsEntry] = []
    @State private var isLoading = true
    @State private var error: String?
    @Environment(\.dismiss) private var dismiss

    // Column widths
    private let positionWidth: CGFloat = 32
    private let logoWidth: CGFloat = 28
    private let teamNameWidth: CGFloat = 120
    private let statColumnWidth: CGFloat = 36
    private let pointsColumnWidth: CGFloat = 40
    private let formPillSize: CGFloat = 24

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()

                if isLoading {
                    ProgressView()
                        .tint(.white)
                } else if let error = error {
                    VStack(spacing: 12) {
                        Image(systemName: "exclamationmark.triangle")
                            .font(.largeTitle)
                            .foregroundStyle(.gray)
                        Text(error)
                            .font(.subheadline)
                            .foregroundStyle(.gray)
                    }
                } else {
                    standingsTable
                }
            }
            .navigationTitle("POSICIONES")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.gray)
                    }
                }
            }
            .toolbarBackground(Color.black, for: .navigationBar)
            .toolbarBackground(.visible, for: .navigationBar)
        }
        .task {
            await loadStandings()
        }
    }

    // MARK: - Standings Table

    private var standingsTable: some View {
        ScrollView(.vertical, showsIndicators: false) {
            VStack(spacing: 0) {
                // Header row
                headerRow

                Divider()
                    .background(Color.gray.opacity(0.3))

                // Team rows
                ForEach(standings) { entry in
                    teamRow(entry: entry)

                    if entry.position < standings.count {
                        Divider()
                            .background(Color.gray.opacity(0.2))
                    }
                }
            }
        }
    }

    // MARK: - Header Row

    private var headerRow: some View {
        HStack(spacing: 0) {
            // Sticky columns (position, logo, name)
            stickyHeader

            // Scrollable stats columns
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 0) {
                    statHeader("PJ")
                    statHeader("G")
                    statHeader("E")
                    statHeader("P")
                    statHeader("GF")
                    statHeader("GC")
                    statHeader("DG")
                    pointsHeader("Pts")
                    formHeader()
                }
            }
        }
        .padding(.vertical, 10)
        .background(Color(white: 0.1))
    }

    private var stickyHeader: some View {
        HStack(spacing: 8) {
            Text("")
                .frame(width: positionWidth)

            Text("Club")
                .font(.caption)
                .fontWeight(.semibold)
                .foregroundStyle(.gray)
                .frame(width: logoWidth + 8 + teamNameWidth, alignment: .leading)
        }
        .padding(.leading, 8)
    }

    private func statHeader(_ title: String) -> some View {
        Text(title)
            .font(.caption)
            .fontWeight(.semibold)
            .foregroundStyle(.gray)
            .frame(width: statColumnWidth)
    }

    private func pointsHeader(_ title: String) -> some View {
        Text(title)
            .font(.caption)
            .fontWeight(.bold)
            .foregroundStyle(.white)
            .frame(width: pointsColumnWidth)
    }

    private func formHeader() -> some View {
        Text("Últimos 5")
            .font(.caption)
            .fontWeight(.semibold)
            .foregroundStyle(.gray)
            .frame(width: CGFloat(5) * (formPillSize + 4) + 8)
    }

    // MARK: - Team Row

    private func teamRow(entry: StandingsEntry) -> some View {
        let isHighlighted = entry.teamId == homeTeamId || entry.teamId == awayTeamId

        return HStack(spacing: 0) {
            // Zone indicator + Sticky columns
            HStack(spacing: 0) {
                zoneIndicator(position: entry.position)

                stickyColumns(entry: entry, isHighlighted: isHighlighted)
            }

            // Scrollable stats
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 0) {
                    statCell(entry.played)
                    statCell(entry.won)
                    statCell(entry.drawn)
                    statCell(entry.lost)
                    statCell(entry.goalsFor)
                    statCell(entry.goalsAgainst)
                    goalDiffCell(entry.goalDiff)
                    pointsCell(entry.points)
                    formCells(entry.formArray)
                }
            }
        }
        .padding(.vertical, 10)
        .background(isHighlighted ? Color.white.opacity(0.08) : Color.clear)
    }

    private func zoneIndicator(position: Int) -> some View {
        let color: Color = {
            switch position {
            case 1...4: return .blue      // Champions League
            case 5...6: return .orange    // Europa League
            case 17...20: return .red     // Relegation
            default: return .clear
            }
        }()

        return Rectangle()
            .fill(color)
            .frame(width: 3)
    }

    private func stickyColumns(entry: StandingsEntry, isHighlighted: Bool) -> some View {
        HStack(spacing: 8) {
            // Position
            Text("\(entry.position)")
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundStyle(isHighlighted ? .white : .gray)
                .frame(width: positionWidth, alignment: .center)

            // Team logo
            if let logoUrl = entry.teamLogo, let url = URL(string: logoUrl) {
                AsyncImage(url: url) { image in
                    image
                        .resizable()
                        .aspectRatio(contentMode: .fit)
                } placeholder: {
                    Image(systemName: "shield.fill")
                        .foregroundStyle(.gray)
                }
                .frame(width: logoWidth, height: logoWidth)
            } else {
                Image(systemName: "shield.fill")
                    .font(.body)
                    .foregroundStyle(.gray)
                    .frame(width: logoWidth, height: logoWidth)
            }

            // Team name
            Text(entry.teamName)
                .font(.subheadline)
                .fontWeight(isHighlighted ? .bold : .regular)
                .foregroundStyle(isHighlighted ? .white : .white.opacity(0.9))
                .lineLimit(1)
                .frame(width: teamNameWidth, alignment: .leading)
        }
        .padding(.leading, 5)
    }

    private func statCell(_ value: Int) -> some View {
        Text("\(value)")
            .font(.subheadline)
            .foregroundStyle(.white.opacity(0.8))
            .frame(width: statColumnWidth)
    }

    private func goalDiffCell(_ value: Int) -> some View {
        Text(value > 0 ? "+\(value)" : "\(value)")
            .font(.subheadline)
            .foregroundStyle(value > 0 ? .green : (value < 0 ? .red : .white.opacity(0.8)))
            .frame(width: statColumnWidth)
    }

    private func pointsCell(_ value: Int) -> some View {
        Text("\(value)")
            .font(.subheadline)
            .fontWeight(.bold)
            .foregroundStyle(.white)
            .frame(width: pointsColumnWidth)
    }

    private func formCells(_ form: [String]) -> some View {
        HStack(spacing: 4) {
            ForEach(0..<5, id: \.self) { index in
                if index < form.count {
                    formPill(result: form[index])
                } else {
                    formPill(result: "-")
                }
            }
        }
        .padding(.horizontal, 4)
    }

    private func formPill(result: String) -> some View {
        let color: Color = {
            switch result {
            case "W": return .green
            case "L": return .red
            case "D": return .gray
            default: return Color(white: 0.2)
            }
        }()

        let icon: String = {
            switch result {
            case "W": return "checkmark"
            case "L": return "xmark"
            case "D": return "minus"
            default: return ""
            }
        }()

        return ZStack {
            Circle()
                .fill(color.opacity(0.2))
                .frame(width: formPillSize, height: formPillSize)

            Circle()
                .stroke(color, lineWidth: 1.5)
                .frame(width: formPillSize, height: formPillSize)

            if !icon.isEmpty {
                Image(systemName: icon)
                    .font(.system(size: 10, weight: .bold))
                    .foregroundStyle(color)
            }
        }
    }

    // MARK: - Load Data

    private func loadStandings() async {
        do {
            let response = try await APIClient.shared.getStandings(leagueId: leagueId)
            standings = response.standings
        } catch {
            self.error = error.localizedDescription
        }
        isLoading = false
    }
}

// MARK: - Preview

#Preview {
    LeagueStandingsView(
        leagueId: 39,
        homeTeamId: 66,  // Aston Villa
        awayTeamId: 65   // Nottingham Forest
    )
    .preferredColorScheme(.dark)
}
