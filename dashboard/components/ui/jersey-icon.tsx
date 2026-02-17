"use client";

interface JerseyIconProps {
  /** Jersey number (1-99) */
  number: number;
  /** Primary fill color or first gradient stop */
  primaryColor?: string;
  /** Second gradient stop (enables vertical gradient when set) */
  secondaryColor?: string;
  /** Number/text color */
  numberColor?: string;
  /** Icon size in pixels */
  size?: number;
  /** Additional CSS class */
  className?: string;
}

/**
 * Flat jersey/kit icon with dynamic number and fill.
 *
 * Supports solid color or vertical gradient fill.
 * Number font matches iOS app scoreboard (Barlow Condensed SemiBold).
 *
 * Usage:
 * ```tsx
 * <JerseyIcon number={9} primaryColor="#E53935" numberColor="#fff" />
 * <JerseyIcon number={10} primaryColor="#1A237E" secondaryColor="#283593" numberColor="#FFD700" />
 * ```
 */
export function JerseyIcon({
  number,
  primaryColor = "#666",
  secondaryColor,
  numberColor = "#000",
  size = 32,
  className,
}: JerseyIconProps) {
  const gradientId = secondaryColor ? `jersey-grad-${number}-${size}` : undefined;
  const fill = gradientId ? `url(#${gradientId})` : primaryColor;
  const label = Math.max(1, Math.min(99, Math.round(number))).toString();
  // Adjust font size: single digit gets larger text
  const fontSize = label.length === 1 ? 38 : 32;

  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 80 80"
      width={size}
      height={size}
      className={className}
      role="img"
      aria-label={`Jersey #${label}`}
    >
      {gradientId && (
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={primaryColor} />
            <stop offset="100%" stopColor={secondaryColor} />
          </linearGradient>
        </defs>
      )}
      {/* T-shirt silhouette */}
      <path
        d="M28 14 C33 6 47 6 52 14 L74 22 L74 36 L62 30 L62 72 L18 72 L18 30 L6 36 L6 22 Z"
        fill="#b7bcc2"
        stroke="#b7bcc2"
        strokeWidth={4}
        strokeLinejoin="round"
      />
      {/* Number */}
      <text
        x="40"
        y="50"
        textAnchor="middle"
        dominantBaseline="central"
        fill={numberColor}
        fontSize={fontSize}
        fontFamily="'Barlow Condensed', 'Arial Narrow', sans-serif"
        fontWeight="600"
        style={{ userSelect: "none" }}
      >
        {label}
      </text>
    </svg>
  );
}
