import type { ReactElement } from "react";

/**
 * Calm bay at dusk, drawn in ink on parchment: a soft graded sky with a peach
 * horizon glow, a headland silhouette, still water with a warm reflection, and
 * two moored sailboats. Soft washes + gradients do most of the work so it reads
 * as one blended scene rather than a pile of strokes.
 *
 * variant="hero"  — full, edge-to-edge under the headline
 * variant="strip" — a framed horizon band shown above the steps
 */

const INK = "#1b1810";

function mulberry32(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rand = mulberry32(9071);
const jit = (n: number) => (rand() - 0.5) * n;

// wispy high clouds — a few long soft strokes leaning right
const wisps: ReactElement[] = [];
for (let i = 0; i < 5; i++) {
  const y = 58 + i * 24 + jit(8);
  const x = 110 + i * 46;
  const len = 340 + rand() * 380;
  wisps.push(
    <path
      key={`w${i}`}
      d={`M${x} ${y} q ${len * 0.3} ${jit(8) - 5} ${len * 0.6} ${jit(5)} t ${len * 0.4} ${jit(7)}`}
      stroke="#6b6675"
      strokeWidth={2.4}
      strokeLinecap="round"
      fill="none"
      opacity={0.26 - i * 0.03}
    />,
  );
}

// broken cloud band just above the horizon, right half
const band: ReactElement[] = [];
for (let i = 0; i < 9; i++) {
  const x = 540 + i * 70 + jit(14);
  const y = 300 - jit(10);
  const len = 28 + rand() * 46;
  band.push(
    <path
      key={`b${i}`}
      d={`M${x} ${y} q ${len / 2} ${-5 - rand() * 6} ${len} 0`}
      stroke="#5c5866"
      strokeWidth={5}
      strokeLinecap="round"
      fill="none"
      opacity={0.38}
    />,
  );
}

// still water — a handful of long gentle ripples, fading down
const ripples: ReactElement[] = [];
for (let i = 0, y = 340; y < 545; i++, y += 15 + rand() * 8) {
  const d = 4 + rand() * 6;
  ripples.push(
    <path
      key={`r${i}`}
      d={`M60 ${y} C 360 ${y - d} 740 ${y + d} 1150 ${y + jit(4)}`}
      stroke={INK}
      strokeWidth={0.8}
      fill="none"
      opacity={Math.max(0.05, 0.3 - (y - 340) / 950)}
    />,
  );
}

const headland =
  "M0 250 C 120 252 210 264 296 246 C 338 237 366 206 408 212 " +
  "C 452 218 470 250 512 258 C 548 265 566 300 604 306 L 604 322 L 0 322 Z";

// tree bumps along the point
const trees: ReactElement[] = [];
for (let i = 0; i < 7; i++) {
  const x = 466 + i * 20 + jit(5);
  trees.push(<path key={`t${i}`} d={`M${x - 8} 306 q 8 -13 16 0 z`} fill={INK} opacity={0.92} />);
}

function Boat({ x, y, s }: { x: number; y: number; s: number }) {
  return (
    <g transform={`translate(${x} ${y}) scale(${s})`}>
      <path d="M-10 0 Q 0 6 10 0" stroke={INK} strokeWidth={1.6} fill="none" />
      <line x1="0" y1="0" x2="0" y2="-22" stroke={INK} strokeWidth={1.4} />
      <path d="M0 -21 L 7 -4 L 0 -4 Z" fill={INK} opacity={0.5} />
    </g>
  );
}

interface Props {
  variant?: "hero" | "strip";
}

export function CoastScene({ variant = "hero" }: Props) {
  const viewBox = variant === "hero" ? "0 0 1200 560" : "0 150 1200 250";
  return (
    <svg
      className={variant === "hero" ? "hero-scene" : "coast-strip"}
      viewBox={viewBox}
      preserveAspectRatio="xMidYMid slice"
      role="img"
      aria-label="Ink illustration of a calm bay at dusk with a headland and two moored sailboats."
    >
      <defs>
        <linearGradient id="cs-sky" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#cdc6d2" stopOpacity="0.5" />
          <stop offset="0.55" stopColor="#e7ddd0" stopOpacity="0.26" />
          <stop offset="0.82" stopColor="#edbf97" stopOpacity="0.7" />
          <stop offset="0.93" stopColor="#f4d3a6" stopOpacity="0.95" />
          <stop offset="1" stopColor="#efe6d4" stopOpacity="0.18" />
        </linearGradient>
        <linearGradient id="cs-sea" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#8b8aa2" stopOpacity="0.42" />
          <stop offset="0.4" stopColor="#c2ab8d" stopOpacity="0.28" />
          <stop offset="0.75" stopColor="#dcc6a1" stopOpacity="0.24" />
          <stop offset="1" stopColor="#cbc4d0" stopOpacity="0.3" />
        </linearGradient>
        <linearGradient id="cs-refl" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#f2cf9c" stopOpacity="0.7" />
          <stop offset="1" stopColor="#f2cf9c" stopOpacity="0" />
        </linearGradient>
        <filter id="cs-soft" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="22" />
        </filter>
      </defs>

      <rect x="0" y="0" width="1200" height="322" fill="url(#cs-sky)" />
      <ellipse
        cx="650"
        cy="322"
        rx="720"
        ry="58"
        fill="#edbf97"
        opacity="0.55"
        filter="url(#cs-soft)"
      />
      <rect x="0" y="322" width="1200" height="240" fill="url(#cs-sea)" />
      <rect x="700" y="322" width="260" height="210" fill="url(#cs-refl)" opacity="0.5" />

      {wisps}
      {band}

      <line x1="0" y1="316" x2="1200" y2="316" stroke={INK} strokeWidth="1.1" opacity="0.55" />

      {ripples}

      <path d={headland} fill={INK} opacity="0.92" />
      {trees}

      <Boat x={1010} y={314} s={1} />
      <Boat x={868} y={315} s={0.82} />
      <line x1="1010" y1="316" x2="1010" y2="333" stroke={INK} strokeWidth="1" opacity="0.22" />
      <line x1="868" y1="316" x2="868" y2="330" stroke={INK} strokeWidth="1" opacity="0.2" />
    </svg>
  );
}
