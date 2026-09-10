import type { ReactElement } from "react";

/**
 * Pen-and-ink illustration of a Nā Pali–style coast overlook: a fluted ridge
 * wall on the right, a green valley opening to the sea, a scalloped cloud bank
 * on the horizon, and a grassy foreground bluff. Hatching is generated from a
 * fixed seed so it never flickers between renders.
 */

const INK = "#1b1810";
const W = 1200;
const H = 640;

function mulberry32(seed: number): () => number {
  return () => {
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const rand = mulberry32(20260910);
const jit = (n: number) => (rand() - 0.5) * n;

// -- ocean: wavering horizontal hatch, fading downward ----------------------
const ocean: ReactElement[] = [];
for (let i = 0, y = 252; y < 452; i++, y += 7 + rand() * 3) {
  const base = y + jit(2);
  const dip = 3 + rand() * 5;
  ocean.push(
    <path
      key={`oc${i}`}
      d={`M46 ${base} C 340 ${base - dip} 620 ${base + dip} 1150 ${base + jit(3)}`}
      stroke={INK}
      strokeWidth={y < 300 ? 1 : 0.8}
      fill="none"
      opacity={Math.max(0.08, 0.5 - (y - 252) / 520)}
    />,
  );
}

// -- scalloped cloud bank on the horizon -----------------------------------
let cx = 58;
let cloudPath = `M${cx} 276`;
while (cx < 1150) {
  const w = 46 + rand() * 42;
  const h = 15 + rand() * 20;
  cloudPath += ` q ${w / 2} ${-h} ${w} 0`;
  cx += w;
}
cloudPath += ` L ${cx} 288 L 58 288 Z`;

const tealPuffs: ReactElement[] = [
  [250, 250, 46, 15],
  [560, 244, 60, 18],
  [910, 252, 40, 13],
].map(([px, py, rx, ry], i) => (
  <ellipse
    key={`tp${i}`}
    cx={px}
    cy={py}
    rx={rx}
    ry={ry}
    fill="var(--teal)"
    stroke={INK}
    strokeWidth={1.4}
    opacity={0.9}
  />
));

// -- right ridge: jagged crest + downward "fluting" strokes ---------------
const crest: Array<[number, number]> = [
  [1200, 92],
  [1168, 118],
  [1150, 90],
  [1120, 150],
  [1100, 114],
  [1074, 176],
  [1050, 136],
  [1028, 202],
  [1002, 158],
  [982, 228],
  [956, 184],
  [938, 254],
  [912, 212],
  [892, 290],
  [866, 250],
  [848, 322],
  [822, 288],
  [806, 358],
  [790, 338],
];
const ridgePath =
  `M1200 92 ${crest.map(([x, y]) => `L${x} ${y}`).join(" ")}` +
  ` L790 340 L800 470 L912 560 L1062 624 L1200 624 Z`;

const flutes: ReactElement[] = [];
crest.forEach(([x, y], i) => {
  for (let s = 0; s < 3; s++) {
    const sx = x + s * 6 - 6 + jit(4);
    const len = 60 + (624 - y) * (0.32 + rand() * 0.26);
    const ex = sx - 12 - rand() * 24;
    const ey = Math.min(y + len, 620);
    flutes.push(
      <path
        key={`fl${i}-${s}`}
        d={`M${sx} ${y + 4} Q ${sx - 6} ${(y + ey) / 2} ${ex} ${ey}`}
        stroke={INK}
        strokeWidth={0.9}
        fill="none"
        opacity={0.55}
      />,
    );
  }
});

// -- left foreground bluff: sweeping curve + contour hatch + bush crest ---
const seg1: Array<[number, number]> = [
  [0, 150],
  [90, 235],
  [210, 250],
  [330, 214],
];
const seg2: Array<[number, number]> = [
  [330, 214],
  [398, 194],
  [442, 250],
  [474, 306],
];
function cubic(t: number, p: Array<[number, number]>): [number, number] {
  const u = 1 - t;
  const b = [u * u * u, 3 * u * u * t, 3 * u * t * t, t * t * t];
  return [
    b[0] * p[0][0] + b[1] * p[1][0] + b[2] * p[2][0] + b[3] * p[3][0],
    b[0] * p[0][1] + b[1] * p[1][1] + b[2] * p[2][1] + b[3] * p[3][1],
  ];
}
const bluffTop =
  "M0 150 C 90 235 210 250 330 214 C 398 194 442 250 474 306";
const bluffPath = `${bluffTop} L 474 640 L 0 640 Z`;

const contours: ReactElement[] = [];
for (let k = 1; k <= 8; k++) {
  const dy = k * 32;
  contours.push(
    <path
      key={`ct${k}`}
      d={`M0 ${150 + dy} C ${90 + jit(10)} ${235 + dy} ${210 + jit(12)} ${250 + dy} ${330 + jit(8)} ${214 + dy} C ${398 + jit(8)} ${194 + dy} ${442 + jit(8)} ${250 + dy} ${474 - k * 5} ${306 + dy}`}
      stroke={INK}
      strokeWidth={0.8}
      fill="none"
      opacity={0.34}
    />,
  );
}

const bushes: ReactElement[] = [];
for (let b = 0; b < 12; b++) {
  const [x, y] =
    b < 7 ? cubic(b / 7, seg1) : cubic((b - 7) / 5, seg2);
  const bx = x + jit(10);
  const by = y - 4 + jit(4);
  bushes.push(
    <path
      key={`bs${b}`}
      d={`M${bx - 12} ${by} q 5 -15 11 -3 q 6 -17 13 -2 q 6 -13 11 2 q -7 8 -16 5 q -11 4 -17 -3 z`}
      fill="var(--cream)"
      stroke={INK}
      strokeWidth={1}
    />,
  );
}

// -- valley floor: rolling top edge + stipple vegetation -----------------
const valleyPath =
  "M0 300 C 150 272 320 296 460 300 C 600 304 700 330 786 392 " +
  "L 786 468 C 700 440 560 468 430 468 C 300 468 150 452 0 470 Z";
const stipple: ReactElement[] = [];
for (let i = 0; i < 90; i++) {
  const x = 30 + rand() * 720;
  const y = 320 + rand() * 130;
  if (x > 470 + (y - 320) * 2.4) continue; // stay left of the ridge base
  stipple.push(
    <path
      key={`st${i}`}
      d={`M${x} ${y} l ${2 + rand() * 3} ${-3 - rand() * 3} l ${2 + rand() * 3} ${3 + rand() * 3}`}
      stroke={INK}
      strokeWidth={0.8}
      fill="none"
      opacity={0.45}
    />,
  );
}

// -- foreground rim (the eroded overlook edge) + grass tufts -------------
const rimPath = "M0 540 C 110 556 230 596 336 640 L 0 640 Z";
const grass: ReactElement[] = [];
for (let i = 0; i < 14; i++) {
  const t = i / 14;
  const x = 8 + t * 320 + jit(10);
  const y = 542 + t * 92 + jit(6);
  grass.push(
    <path
      key={`gr${i}`}
      d={`M${x} ${y} q ${jit(6)} -14 ${2 + jit(4)} -22`}
      stroke={INK}
      strokeWidth={1}
      fill="none"
    />,
  );
}

// -- birds -------------------------------------------------------------------
const birds: ReactElement[] = [
  [360, 150],
  [420, 132],
  [500, 168],
].map(([x, y], i) => (
  <path
    key={`bd${i}`}
    d={`M${x} ${y} q 7 -7 14 0 q 7 -7 14 0`}
    stroke={INK}
    strokeWidth={1.2}
    fill="none"
  />
));

export function HeroScene() {
  return (
    <svg
      className="hero-scene"
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMax slice"
      role="img"
      aria-label="Ink illustration of a coastal overlook: a fluted ridge, a green valley, and the open sea."
    >
      <defs>
        <filter id="ob-soft" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="20" />
        </filter>
      </defs>

      {/* warm horizon glow */}
      <ellipse
        cx="600"
        cy="262"
        rx="740"
        ry="52"
        fill="var(--peach)"
        opacity="0.5"
        filter="url(#ob-soft)"
      />

      {/* sea */}
      <path
        d="M0 250 L1200 250 L1200 470 C 900 452 300 452 0 480 Z"
        fill="#7fa9b8"
        opacity="0.13"
      />
      <g>{ocean}</g>

      {/* clouds */}
      <path d={cloudPath} fill="var(--cream)" stroke={INK} strokeWidth="1.4" />
      {tealPuffs}
      <path d={cloudPath} fill="none" stroke={INK} strokeWidth="1.4" />

      {birds}

      {/* valley */}
      <path d={valleyPath} fill="var(--cream)" />
      <path d={valleyPath} fill="#8fae72" opacity="0.16" />
      <path d={valleyPath} fill="none" stroke={INK} strokeWidth="1.2" />
      <g>{stipple}</g>

      {/* right ridge */}
      <path d={ridgePath} fill="var(--cream)" />
      <path d={ridgePath} fill="#6f8f6a" opacity="0.18" />
      <g>{flutes}</g>
      <path d={ridgePath} fill="none" stroke={INK} strokeWidth="1.8" />

      {/* left bluff */}
      <path d={bluffPath} fill="var(--cream)" />
      <path d={bluffPath} fill="#caa96f" opacity="0.2" />
      <g>{contours}</g>
      <path d={bluffTop} fill="none" stroke={INK} strokeWidth="1.8" />
      {bushes}

      {/* foreground rim */}
      <path d={rimPath} fill="#2a251b" opacity="0.9" />
      <path d={rimPath} fill="none" stroke={INK} strokeWidth="1.6" />
      <g>{grass}</g>
    </svg>
  );
}
