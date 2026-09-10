# Screenshots

Used in the top-level `README.md`. Regenerate after a UI change:

```bash
# with the stack running (docker compose up -d)
npm --prefix /tmp/ob-shots init -y && npm --prefix /tmp/ob-shots i playwright
npx --prefix /tmp/ob-shots playwright install chromium
node - <<'JS'
import { chromium } from "playwright";
const OUT = "docs/screenshots";
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: 1320, height: 860 }, deviceScaleFactor: 2 });
await p.goto("http://localhost:3000", { waitUntil: "networkidle" });
await p.waitForTimeout(700);
await p.screenshot({ path: `${OUT}/hero.png` });
await p.locator("#how").screenshot({ path: `${OUT}/how-it-works.png` });
await p.locator("#tool .tool-card").screenshot({ path: `${OUT}/redact.png` });
await b.close();
JS
```

`hero.png` is the viewport at the top of the page; the other two clip a section.
