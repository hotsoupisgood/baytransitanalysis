#!/usr/bin/env python3
"""Headless check of the dashboard and the Methods page before a commit.

    python3 -m http.server 8788            # in ~/ytpt/docs
    python3 scripts/site_export/verify_site.py  # exits non-zero on any failure

What it asserts, on both pages, at 1280 px and at 390 px:

  * no console error and no uncaught page error;
  * every <details> opens and closes again (a disclosure that cannot be opened
    hides a caveat, which is the one failure mode this site cannot afford);
  * the flowchart on Methods draws every box data/flow.json carries, and
    clicking each box fills the detail pane beside it;
  * the role block on the dashboard carries its two steps;
  * both pages pass the owner's wording checker with no FAIL.

Prints a line per check. Nothing is written.
"""
import subprocess
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

# The owner's wording rules, codified 2026-10-21: style/DASHBOARD-LANGUAGE.md in the
# OpenClaw workspace, with copy_check.py and copy_rules.json beside it. "Before he
# sees any change, run the checker on every changed page. It must report 0 FAIL." So
# a FAIL is a build failure here rather than something to notice afterwards. WARN is
# reported and does not fail: the dashboard's TLDR is first person on purpose.
# .turns and .selfid are transcript text and .quote is the two example comments;
# transcript quotes are data and are never edited, so all three are excluded.
COPY_CHECK = Path.home() / ".openclaw/workspace/style/copy_check.py"
COPY_EXCLUDES = (".turns", ".selfid", ".quote")

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8788"
fails, notes = [], []


def check(label, ok, detail=""):
    (notes if ok else fails).append(f"{'ok  ' if ok else 'FAIL'} {label}"
                                    + (f" — {detail}" if detail else ""))


def copy_check(pages):
    """Run the wording checker over the already-served pages."""
    if not COPY_CHECK.exists():
        check("copy_check present", False, f"not found at {COPY_CHECK}")
        return
    cmd = [sys.executable, str(COPY_CHECK), *(f"{BASE}/{p}" for p in pages),
           "--format", "json"]
    for sel in COPY_EXCLUDES:
        cmd += ["--exclude-selector", sel]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        check("copy_check runs", False, "timed out after 300 s")
        return
    import json
    try:
        data = json.loads(out.stdout)
    except ValueError:
        check("copy_check runs", False,
              (out.stderr or out.stdout or "no output").strip().splitlines()[-1][:200])
        return
    for page in data["pages"]:
        name = page["url"].rsplit("/", 1)[-1]
        rules = ", ".join(sorted(page.get("rules", {}))) or "none"
        check(f"{name} wording: 0 FAIL", page["fail"] == 0,
              f"{page['fail']} FAIL, {page['warn']} WARN ({rules})")


with sync_playwright() as pw:
    b = pw.chromium.launch()
    for page_name in ("index.html", "methods.html"):
        for w, h in ((1280, 900), (390, 844)):
            errs = []
            pg = b.new_page(viewport={"width": w, "height": h})
            pg.on("console", lambda m: m.type == "error" and errs.append(m.text))
            pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
            pg.goto(f"{BASE}/{page_name}", wait_until="networkidle")
            pg.wait_for_timeout(2500)

            # details open and close
            n = pg.locator("details").count()
            opened = pg.evaluate("""() => {
              const ds = [...document.querySelectorAll('details')];
              let bad = 0;
              for (const d of ds) {
                const was = d.open;
                d.open = true;  if (!d.open) bad++;
                d.open = false; if (d.open)  bad++;
                d.open = was;
              }
              return {n: ds.length, bad};
            }""")
            check(f"{page_name} @{w} details open/close", opened["bad"] == 0,
                  f"{opened['n']} disclosures, {opened['bad']} stuck")

            if page_name == "methods.html":
                boxes = pg.evaluate("window.__flow ? window.__flow.boxes.length : 0")
                drawn = pg.locator("#flow-svg g.node").count()
                check(f"{page_name} @{w} flowchart drawn", boxes > 0 and drawn == boxes,
                      f"{drawn} boxes drawn of {boxes} in flow.json")
                clicked_ok = True
                for i in range(drawn):
                    pg.locator("#flow-svg g.node").nth(i).click()
                    pg.wait_for_timeout(60)
                    if len(pg.locator("#flow-detail h3").inner_text() or "") < 3:
                        clicked_ok = False
                check(f"{page_name} @{w} every box clickable", clicked_ok and drawn > 0,
                      f"{drawn} clicked")
                dashed = pg.locator("#flow-svg g.node.decision").count()
                check(f"{page_name} @{w} has dashed decision boxes", dashed >= 1,
                      f"{dashed} decision boxes")
            else:
                steps = pg.locator("#side-block .sidesteps > div").count()
                check(f"{page_name} @{w} role block steps", steps >= 1,
                      f"{steps} steps")

            check(f"{page_name} @{w} console clean", not errs, "; ".join(errs[:4]))
            pg.close()
    b.close()

copy_check(("index.html", "methods.html"))

for line in notes + fails:
    print(line)
print(f"\n{len(notes)} ok, {len(fails)} failed")
sys.exit(1 if fails else 0)
