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
  numberColor = "rgba(255,255,255,0.5)",
  size = 32,
  className,
}: JerseyIconProps) {
  const gradientId = secondaryColor ? `jersey-grad-${number}-${size}` : undefined;
  const fill = gradientId ? `url(#${gradientId})` : primaryColor;
  const label = Math.max(1, Math.min(99, Math.round(number))).toString();
  const fontSize = 46;

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
        d="M28 14 C33 6 47 6 52 14 L74 18 Q76 27 74 36 Q68 36 62 40 L62 66 Q62 72 56 72 Q48 76 40 76 Q32 76 24 72 Q18 72 18 66 L18 40 Q12 36 6 36 Q4 27 6 18 Z"
        fill="rgba(255,255,255,0.15)"
        stroke="none"
        filter="url(#shirt-shadow)"
      />
      <defs>
        <filter id="shirt-shadow" x="-5%" y="-5%" width="110%" height="110%">
          <feDropShadow dx="0" dy="0.5" stdDeviation="0.5" floodColor="#000" floodOpacity="0.4" />
        </filter>
        <filter id="num-shadow" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="1" stdDeviation="1.5" floodColor="#000" floodOpacity="0.6" />
        </filter>
      </defs>
      {/* Number */}
      <text
        x="40"
        y="42"
        textAnchor="middle"
        dominantBaseline="central"
        fill={numberColor}
        fontSize={fontSize}
        fontFamily="'Barlow Condensed', 'Arial Narrow', sans-serif"
        fontWeight="400"
        filter="url(#num-shadow)"
        style={{ userSelect: "none" }}
      >
        {label}
      </text>
    </svg>
  );
}
