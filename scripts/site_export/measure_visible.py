#!/usr/bin/env python3
"""Count the words a reader actually sees, and screenshot the page.

Visible = document.innerText of <main>, which a rendering engine computes after
layout: text inside a closed <details> is not rendered and so is not counted.
That is the quantity the 2026-10-19 concision pass is measured against, panel by
panel, so "half the words" means half the words on screen rather than half the
words in the file. Run it against a locally served copy:

    python3 docs/measure_visible.py http://localhost:8788/index.html
    python3 docs/measure_visible.py http://localhost:8788/index.html --shots PREFIX
"""
import json
import re
import sys
from playwright.sync_api import sync_playwright

# Each panel's visible text is read off the element the page renders it into, so
# the per-panel counts follow the markup rather than a guess at where a panel ends.
SECTIONS = {
    "intro": "#intro",
    "tldr": "#tldr",
    "rates": "#p-rates",
    "nimby": "#p-nimby",
    "side-block": "#side-block",
    "stance": "#p-stance",
    "reader": "#reader",
    "methods-footer": "#methods",
}
WORDS = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’%×·\-\.,/()+−–]*")


def count(txt):
    return len(WORDS.findall(txt or ""))


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8788/index.html"
    shots = sys.argv[sys.argv.index("--shots") + 1] if "--shots" in sys.argv else None
    out = {"url": url, "console": [], "sections": {}}
    with sync_playwright() as pw:
        b = pw.chromium.launch()
        pg = b.new_page(viewport={"width": 1280, "height": 900})
        pg.on("console", lambda m: m.type in ("error", "warning") and out["console"].append(
            f"{m.type}: {m.text}"))
        pg.on("pageerror", lambda e: out["console"].append(f"pageerror: {e}"))
        pg.goto(url, wait_until="networkidle")
        pg.wait_for_timeout(1500)
        out["total"] = count(pg.inner_text("main"))
        for name, sel in SECTIONS.items():
            n = pg.locator(sel)
            out["sections"][name] = count(n.inner_text()) if n.count() else None
        out["details"] = pg.eval_on_selector_all("main details", "els => els.length")
        out["svgs"] = pg.eval_on_selector_all(
            "main svg", "els => els.map(e => e.id || e.getAttribute('role')).filter(Boolean)")
        # One open/close cycle on every disclosure, so a broken <details> shows up
        # as a count that does not come back to where it started.
        pg.eval_on_selector_all("main details", "els => els.forEach(e => e.open = true)")
        pg.wait_for_timeout(400)
        out["total_all_open"] = count(pg.inner_text("main"))
        pg.eval_on_selector_all("main details", "els => els.forEach(e => e.open = false)")
        pg.wait_for_timeout(400)
        out["total_reclosed"] = count(pg.inner_text("main"))
        if shots:
            pg.wait_for_timeout(300)
            h = pg.evaluate("document.body.scrollHeight")
            out["desktop_height"] = h
            pg.screenshot(path=f"{shots}-desktop.png", full_page=True)
            role = pg.locator("#rdp-panel")
            if role.count():
                role.scroll_into_view_if_needed()
                pg.wait_for_timeout(400)
                role.screenshot(path=f"{shots}-role.png")
            pg.set_viewport_size({"width": 390, "height": 844})
            pg.wait_for_timeout(1200)
            pg.screenshot(path=f"{shots}-phone.png", full_page=True)
            out["phone_height"] = pg.evaluate("document.body.scrollHeight")
        b.close()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
