/**
 * Calm bay at dusk — the hero background.
 *
 * Soft gradient sky and sea, one warm horizon glow, a clean headland
 * silhouette, layered sine-wave water, two sailboats. No randomness, so it
 * renders identically every time.
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
  "M0 400 C 140 396 262 384 360 366 C 430 354 470 376 520 406 " +
  "C 560 430 586 456 628 468 L 628 474 L 0 474 Z";

/** A true sine, sampled finely into a rounded polyline — reads as water. */
function sineWave(baseY: number, wavelength: number, amp: number, phase: number): string {
  const pts: string[] = [];
  for (let x = -40; x <= 1240; x += 9) {
    const y = baseY + amp * Math.sin(((x + phase) / wavelength) * Math.PI * 2);
    pts.push(`${x} ${y.toFixed(2)}`);
  }
  return `M${pts.join(" L ")}`;
}

const WAVE_ROWS = 6;
const WAVES = Array.from({ length: WAVE_ROWS }, (_, i) => {
  const t = i / (WAVE_ROWS - 1); // 0 near the horizon, 1 in the foreground
  const baseY = 494 + t * 250;
  const wavelength = 150 + t * 150;
  const amp = 2 + t * 6;
  const phase = i * 137; // offset each row so crests don't line up
  return (
    <path
      key={`wv${i}`}
      d={sineWave(baseY, wavelength, amp, phase)}
      stroke={INK}
      strokeWidth={0.9 + t * 0.5}
      fill="none"
      strokeLinecap="round"
      strokeLinejoin="round"
      opacity={0.14 + t * 0.08}
    />
  );
});

export function CoastScene() {
  return (
    <svg
      className="hero-bg"
      viewBox="0 0 1200 800"
      preserveAspectRatio="xMidYMax slice"
      role="img"
      aria-label="Ink illustration of a calm bay at dusk with a headland and two moored sailboats."
    >
      <defs>
        <linearGradient id="cs-sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#f4efe2" />
          <stop offset="0.4" stopColor="#e9e0d2" />
          <stop offset="0.66" stopColor="#eccaa6" />
          <stop offset="0.84" stopColor="#f4cd9f" />
          <stop offset="0.94" stopColor="#f9deb7" />
          <stop offset="1" stopColor="#e7d9c8" />
        </linearGradient>
        <linearGradient id="cs-sea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#9a94a6" stopOpacity="0.5" />
          <stop offset="0.3" stopColor="#c7ac8c" stopOpacity="0.36" />
          <stop offset="0.6" stopColor="#dcc39c" stopOpacity="0.3" />
          <stop offset="1" stopColor="#cfc8d2" stopOpacity="0.38" />
        </linearGradient>
        <linearGradient id="cs-refl" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#f4d3a8" stopOpacity="0.6" />
          <stop offset="1" stopColor="#f4d3a8" stopOpacity="0" />
        </linearGradient>
        <filter id="cs-soft" x="-40%" y="-40%" width="180%" height="180%">
          <feGaussianBlur stdDeviation="26" />
        </filter>
        <filter id="cs-cloud" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="9" />
        </filter>
      </defs>

      {/* sky + horizon glow */}
      <rect x="0" y="0" width="1200" height="470" fill="url(#cs-sky)" />
      <ellipse cx="620" cy="470" rx="780" ry="74" fill="#f1caa1" opacity="0.5" filter="url(#cs-soft)" />

      {/* clouds */}
      <g filter="url(#cs-cloud)" fill="#8b8797">
        <ellipse cx="770" cy="410" rx="132" ry="15" opacity="0.32" />
        <ellipse cx="930" cy="396" rx="92" ry="12" opacity="0.28" />
        <ellipse cx="1050" cy="418" rx="70" ry="13" opacity="0.26" />
      </g>
      <path
        d="M120 150 C 360 136 620 150 900 132"
        stroke="#8f8b99"
        strokeWidth="3"
        strokeLinecap="round"
        fill="none"
        opacity="0.15"
      />
      <path
        d="M240 208 C 430 200 600 210 780 198"
        stroke="#8f8b99"
        strokeWidth="2.4"
        strokeLinecap="round"
        fill="none"
        opacity="0.11"
      />

      {/* sea */}
      <rect x="0" y="470" width="1200" height="330" fill="url(#cs-sea)" />
      <rect x="770" y="470" width="230" height="220" fill="url(#cs-refl)" opacity="0.5" />
      <line x1="0" y1="470" x2="1200" y2="470" stroke={INK} strokeWidth="1.1" opacity="0.45" />
      {WAVES}

      {/* headland */}
      <path d={HEADLAND} fill={INK} opacity="0.92" />
      <ellipse cx="596" cy="464" rx="18" ry="9" fill={INK} opacity="0.92" />
      <ellipse cx="628" cy="468" rx="12" ry="7" fill={INK} opacity="0.92" />

      {/* boats */}
      <Boat x={1012} y={468} s={1} />
      <Boat x={872} y={469} s={0.82} />
      <line x1="1012" y1="470" x2="1012" y2="488" stroke={INK} strokeWidth="1" opacity="0.2" />
      <line x1="872" y1="470" x2="872" y2="485" stroke={INK} strokeWidth="1" opacity="0.18" />
    </svg>
  );
}
