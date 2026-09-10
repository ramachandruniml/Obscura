/**
 * Calm bay at dusk — the hero backdrop and the "how it works" band.
 *
 * Deliberately minimal: soft gradient sky and sea, one warm horizon glow, a
 * clean headland silhouette, a few precise ripple lines, two sailboats. No
 * randomness, so it renders identically every time.
 *
 * variant="hero"  — absolute, fills the hero section behind the headline
 * variant="strip" — a framed horizon band above the steps
 */

const INK = "#1b1810";

function Boat({ x, y, s }: { x: number; y: number; s: number }) {
  return (
    <g
      transform={`translate(${x} ${y}) scale(${s})`}
      fill="none"
      stroke={INK}
      strokeWidth={1.6}
      strokeLinecap="round"
    >
      <path d="M-11 0 Q 0 7 11 0" />
      <line x1="0" y1="0" x2="0" y2="-26" />
      <path d="M0 -25 L 8 -5 L 0 -5 Z" fill={INK} stroke="none" opacity="0.6" />
      <path d="M0 -19 L -6 -6 L 0 -6 Z" fill={INK} stroke="none" opacity="0.32" />
    </g>
  );
}

const HEADLAND =
  "M0 360 C 140 356 262 344 360 326 C 430 314 470 336 520 366 " +
  "C 560 390 586 416 628 428 L 628 434 L 0 434 Z";

/** A smooth continuous wave along the whole width: one crest, then reflected
 *  half-waves ('t') all the way across. */
function wavePath(baseY: number, wavelength: number, amp: number): string {
  const half = wavelength / 2;
  let d = `M-60 ${baseY} q ${half / 2} ${-amp} ${half} 0`;
  const count = Math.ceil(1320 / half);
  for (let i = 0; i < count; i++) d += ` t ${half} 0`;
  return d;
}

const WAVE_ROWS = 9;
const WAVES = Array.from({ length: WAVE_ROWS }, (_, i) => {
  const t = i / (WAVE_ROWS - 1); // 0 near the horizon, 1 in the foreground
  const baseY = 450 + t * 280;
  const wavelength = 64 + t * 104;
  const amp = 1.4 + t * 6.2;
  return (
    <path
      key={`wv${i}`}
      d={wavePath(baseY, wavelength, amp)}
      stroke={INK}
      strokeWidth={0.8 + t * 0.5}
      fill="none"
      strokeLinecap="round"
      opacity={0.09 + t * 0.16}
    />
  );
});

interface Props {
  variant?: "hero" | "strip";
}

export function CoastScene({ variant = "hero" }: Props) {
  const hero = variant === "hero";
  return (
    <svg
      className={hero ? "hero-bg" : "coast-strip"}
      viewBox={hero ? "0 0 1200 760" : "0 250 1200 300"}
      preserveAspectRatio="xMidYMax slice"
      role="img"
      aria-label="Ink illustration of a calm bay at dusk with a headland and two moored sailboats."
    >
      <defs>
        <linearGradient id="cs-sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#f4efe2" />
          <stop offset="0.34" stopColor="#e9e0d2" />
          <stop offset="0.62" stopColor="#eccaa6" />
          <stop offset="0.82" stopColor="#f4cd9f" />
          <stop offset="0.93" stopColor="#f9deb7" />
          <stop offset="1" stopColor="#e7d9c8" />
        </linearGradient>
        <linearGradient id="cs-sea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#9a94a6" stopOpacity="0.55" />
          <stop offset="0.3" stopColor="#c7ac8c" stopOpacity="0.4" />
          <stop offset="0.6" stopColor="#dcc39c" stopOpacity="0.32" />
          <stop offset="1" stopColor="#cfc8d2" stopOpacity="0.4" />
        </linearGradient>
        <linearGradient id="cs-refl" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#f4d3a8" stopOpacity="0.65" />
          <stop offset="1" stopColor="#f4d3a8" stopOpacity="0" />
        </linearGradient>
        <linearGradient id="cs-scrim" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#f4efe2" stopOpacity="0.92" />
          <stop offset="1" stopColor="#f4efe2" stopOpacity="0" />
        </linearGradient>
        <filter id="cs-soft" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="26" />
        </filter>
        <filter id="cs-cloud" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="9" />
        </filter>
      </defs>

      {/* sky + horizon glow */}
      <rect x="0" y="0" width="1200" height="430" fill="url(#cs-sky)" />
      <ellipse
        cx="620"
        cy="430"
        rx="760"
        ry="72"
        fill="#f1caa1"
        opacity="0.5"
        filter="url(#cs-soft)"
      />

      {/* clouds */}
      <g filter="url(#cs-cloud)" fill="#8b8797">
        <ellipse cx="770" cy="374" rx="132" ry="15" opacity="0.34" />
        <ellipse cx="930" cy="360" rx="92" ry="12" opacity="0.3" />
        <ellipse cx="1050" cy="382" rx="70" ry="13" opacity="0.28" />
      </g>
      <path
        d="M120 118 C 360 104 620 118 900 100"
        stroke="#8f8b99"
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
        opacity="0.16"
      />
      <path
        d="M240 176 C 430 168 600 178 780 166"
        stroke="#8f8b99"
        strokeWidth="2.4"
        strokeLinecap="round"
        fill="none"
        opacity="0.12"
      />

      {/* sea */}
      <rect x="0" y="430" width="1200" height="330" fill="url(#cs-sea)" />
      <rect x="770" y="430" width="230" height="210" fill="url(#cs-refl)" opacity="0.55" />
      <line x1="0" y1="430" x2="1200" y2="430" stroke={INK} strokeWidth="1.1" opacity="0.5" />
      {WAVES}

      {/* headland */}
      <path d={HEADLAND} fill={INK} opacity="0.92" />
      <ellipse cx="596" cy="424" rx="18" ry="9" fill={INK} opacity="0.92" />
      <ellipse cx="628" cy="428" rx="12" ry="7" fill={INK} opacity="0.92" />

      {/* boats */}
      <Boat x={1012} y={428} s={1} />
      <Boat x={872} y={429} s={0.82} />
      <line x1="1012" y1="430" x2="1012" y2="448" stroke={INK} strokeWidth="1" opacity="0.22" />
      <line x1="872" y1="430" x2="872" y2="445" stroke={INK} strokeWidth="1" opacity="0.2" />

      {/* top scrim so the headline always sits on a clean field */}
      {hero && <rect x="0" y="0" width="1200" height="300" fill="url(#cs-scrim)" />}
    </svg>
  );
}
