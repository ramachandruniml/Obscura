export function Arrow() {
  return (
    <svg width="18" height="14" viewBox="0 0 18 14" fill="none" aria-hidden="true">
      <path
        d="M1 7h15M11 1l6 6-6 6"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

/** A censor bar over an eye. */
export function BrandMark() {
  return (
    <svg width="26" height="26" viewBox="0 0 26 26" fill="none" aria-hidden="true">
      <circle cx="13" cy="13" r="11" stroke="var(--ink)" strokeWidth="1.8" />
      <circle cx="13" cy="13" r="3.6" fill="var(--ink)" />
      <rect
        x="1.5"
        y="10.4"
        width="23"
        height="5.2"
        rx="2.6"
        fill="var(--teal)"
        stroke="var(--ink)"
        strokeWidth="1.6"
      />
    </svg>
  );
}
