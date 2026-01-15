import SwiftUI

// MARK: - Shimmer Effect Modifier

struct ShimmerModifier: ViewModifier {
    @State private var phase: CGFloat = 0
    let animation: Animation

    init(animation: Animation = .linear(duration: 1.5).repeatForever(autoreverses: false)) {
        self.animation = animation
    }

    func body(content: Content) -> some View {
        content
            .overlay(
                GeometryReader { geo in
                    LinearGradient(
                        gradient: Gradient(colors: [
                            .clear,
                            .white.opacity(0.2),
                            .clear
                        ]),
                        startPoint: .leading,
                        endPoint: .trailing
                    )
                    .frame(width: geo.size.width * 2)
                    .offset(x: -geo.size.width + (geo.size.width * 2 * phase))
                }
            )
            .mask(content)
            .onAppear {
                withAnimation(animation) {
                    phase = 1
                }
            }
    }
}

extension View {
    func shimmer() -> some View {
        modifier(ShimmerModifier())
    }
}

// MARK: - Base Skeleton Shape

struct SkeletonShape: View {
    let width: CGFloat?
    let height: CGFloat

    init(width: CGFloat? = nil, height: CGFloat = 16) {
        self.width = width
        self.height = height
    }

    var body: some View {
        RoundedRectangle(cornerRadius: height / 4)
            .fill(Color(white: 0.15))
            .frame(width: width, height: height)
            .shimmer()
    }
}

// MARK: - Match Header Skeleton

struct MatchHeaderSkeleton: View {
    var body: some View {
        VStack(spacing: 16) {
            HStack(alignment: .top, spacing: 0) {
                // Home team skeleton
                teamColumnSkeleton

                // VS section
                VStack(spacing: 8) {
                    Spacer()
                    SkeletonShape(width: 50, height: 28)
                    SkeletonShape(width: 80, height: 14)
                    Spacer()
                }
                .frame(width: 100)

                // Away team skeleton
                teamColumnSkeleton
            }
            .frame(height: 160)

            // Prediction badge skeleton
            SkeletonShape(width: 120, height: 32)
        }
        .padding(.top, 8)
    }

    private var teamColumnSkeleton: some View {
        VStack(spacing: 6) {
            // Star placeholder
            SkeletonShape(width: 24, height: 24)

            // Logo placeholder
            Circle()
                .fill(Color(white: 0.15))
                .frame(width: 72, height: 72)
                .shimmer()

            // Team name
            SkeletonShape(width: 80, height: 16)

            // Position
            SkeletonShape(width: 60, height: 12)
        }
        .frame(maxWidth: .infinity)
    }
}

// MARK: - Probability Bar Skeleton

struct ProbabilityBarSkeleton: View {
    var body: some View {
        VStack(spacing: 16) {
            // Bar
            SkeletonShape(height: 16)

            // Labels
            HStack {
                probabilityLabelSkeleton
                Spacer()
                probabilityLabelSkeleton
                Spacer()
                probabilityLabelSkeleton
            }
        }
    }

    private var probabilityLabelSkeleton: some View {
        VStack(spacing: 4) {
            Circle()
                .fill(Color(white: 0.15))
                .frame(width: 10, height: 10)
                .shimmer()
            SkeletonShape(width: 40, height: 12)
            SkeletonShape(width: 50, height: 24)
        }
    }
}

// MARK: - Odds Cards Skeleton

struct OddsCardsSkeleton: View {
    var body: some View {
        HStack(spacing: 12) {
            oddsCardSkeleton
            oddsCardSkeleton
            oddsCardSkeleton
        }
    }

    private var oddsCardSkeleton: some View {
        VStack(spacing: 10) {
            SkeletonShape(width: 50, height: 16)

            VStack(spacing: 2) {
                SkeletonShape(width: 60, height: 10)
                SkeletonShape(width: 50, height: 28)
            }

            VStack(spacing: 2) {
                SkeletonShape(width: 50, height: 10)
                SkeletonShape(width: 50, height: 28)
            }
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
}

// MARK: - Form Table Skeleton

struct FormTableSkeleton: View {
    var body: some View {
        VStack(spacing: 0) {
            formRowSkeleton

            Divider()
                .background(Color.gray.opacity(0.3))

            formRowSkeleton

            // Chevron
            SkeletonShape(width: 16, height: 8)
                .padding(.vertical, 8)
        }
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    private var formRowSkeleton: some View {
        HStack(spacing: 0) {
            SkeletonShape(width: 30, height: 16)
                .frame(width: 36, alignment: .leading)

            Circle()
                .fill(Color(white: 0.15))
                .frame(width: 32, height: 32)
                .shimmer()

            Spacer()

            // Form pills
            HStack(spacing: 6) {
                ForEach(0..<5, id: \.self) { _ in
                    RoundedRectangle(cornerRadius: 6)
                        .fill(Color(white: 0.15))
                        .frame(width: 28, height: 28)
                        .shimmer()
                }
            }

            Spacer()

            SkeletonShape(width: 56, height: 24)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 14)
    }
}

// MARK: - Timeline Skeleton

struct TimelineSkeleton: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            SkeletonShape(width: 150, height: 20)

            // Timeline items
            ForEach(0..<4, id: \.self) { _ in
                HStack(spacing: 12) {
                    SkeletonShape(width: 40, height: 14)
                    Circle()
                        .fill(Color(white: 0.15))
                        .frame(width: 8, height: 8)
                        .shimmer()
                    SkeletonShape(height: 14)
                }
            }
        }
        .padding(16)
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
}

// MARK: - Narrative Skeleton

struct NarrativeSkeleton: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            // Title
            SkeletonShape(width: 200, height: 24)

            // Body lines
            VStack(alignment: .leading, spacing: 8) {
                SkeletonShape(height: 14)
                SkeletonShape(height: 14)
                SkeletonShape(width: 280, height: 14)
                SkeletonShape(height: 14)
                SkeletonShape(width: 200, height: 14)
            }

            // Key factors
            VStack(alignment: .leading, spacing: 8) {
                SkeletonShape(width: 100, height: 16)
                ForEach(0..<3, id: \.self) { _ in
                    HStack(spacing: 8) {
                        Circle()
                            .fill(Color(white: 0.15))
                            .frame(width: 6, height: 6)
                            .shimmer()
                        SkeletonShape(width: 200, height: 12)
                    }
                }
            }
        }
        .padding(16)
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
}

// MARK: - Stats Table Skeleton

struct StatsTableSkeleton: View {
    var body: some View {
        VStack(spacing: 12) {
            // Header
            SkeletonShape(width: 120, height: 20)

            // Stat rows
            ForEach(0..<6, id: \.self) { _ in
                HStack {
                    SkeletonShape(width: 40, height: 16)
                    Spacer()
                    SkeletonShape(width: 80, height: 12)
                    Spacer()
                    SkeletonShape(width: 40, height: 16)
                }
            }
        }
        .padding(16)
        .background(Color(white: 0.1))
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }
}

// MARK: - Full Match Detail Skeleton

struct MatchDetailSkeleton: View {
    var body: some View {
        VStack(spacing: 24) {
            MatchHeaderSkeleton()
            ProbabilityBarSkeleton()
            OddsCardsSkeleton()
            FormTableSkeleton()
        }
        .padding(.horizontal, 16)
    }
}

// MARK: - Preview

#Preview {
    ScrollView {
        VStack(spacing: 32) {
            MatchDetailSkeleton()

            Divider()

            TimelineSkeleton()

            NarrativeSkeleton()

            StatsTableSkeleton()
        }
        .padding()
    }
    .background(Color.black)
    .preferredColorScheme(.dark)
}
