#!/usr/bin/env python3
"""Screenshot the dashboard at desktop and phone width, plus the role panel alone.

Serve the site first, then run against that URL:

    python3 -m http.server 8788            # in ~/ytpt/docs
    python3 scripts/site_export/shoot_screenshots.py <url> <prefix>

Writes five files: the full-page desktop render cut in half (it is past 6,000 px,
which is too tall for one readable file), the full-page phone render at 390 px,
the role dot-plot panel on its own, and the Methods page's flowchart with its
detail pane. Console errors and page errors are printed rather than swallowed —
a screenshot of a broken page should say so.
"""
import sys
from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8788/index.html"
P = sys.argv[2] if len(sys.argv) > 2 else "/tmp/thtrans-shots/shot"

errs = []
with sync_playwright() as pw:
    b = pw.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 900})
    pg.on("console", lambda m: m.type in ("error", "warning") and errs.append(f"{m.type}: {m.text}"))
    pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
    pg.goto(URL, wait_until="networkidle")
    pg.wait_for_timeout(2000)
    # Every panel figure is loading="lazy", so a full-page shot taken straight away
    # catches the ones below the fold undecoded. Load them all, then wait for decode.
    pg.eval_on_selector_all("img", "els => els.forEach(e => e.loading = 'eager')")
    pg.evaluate("Promise.all([...document.images].map(i => i.complete ? 1 : i.decode().catch(() => 1)))")
    pg.wait_for_timeout(1500)
    # The desktop render is cut at a pixel boundary rather than scaled down.
    h = pg.evaluate("document.body.scrollHeight")
    half = h // 2
    pg.screenshot(path=f"{P}-desktop-1.png", full_page=True,
                  clip={"x": 0, "y": 0, "width": 1280, "height": half})
    pg.screenshot(path=f"{P}-desktop-2.png", full_page=True,
                  clip={"x": 0, "y": half, "width": 1280, "height": h - half})
    r = pg.locator("#rdp-panel")
    r.scroll_into_view_if_needed()
    pg.wait_for_timeout(600)
    r.screenshot(path=f"{P}-role.png")
    pg.set_viewport_size({"width": 390, "height": 844})
    pg.wait_for_timeout(1500)
    pg.evaluate("window.scrollTo(0,0)")
    pg.wait_for_timeout(400)
    pg.screenshot(path=f"{P}-phone.png", full_page=True)
    ph = pg.evaluate("document.body.scrollHeight")
    pg.close()

    # The flowchart on the Methods page, with the pane beside it: the chart and
    # the detail it fills in are one thing, so the shot covers both panels.
    mg = b.new_page(viewport={"width": 1280, "height": 900})
    mg.on("console", lambda m: m.type in ("error", "warning") and errs.append(f"methods {m.type}: {m.text}"))
    mg.on("pageerror", lambda e: errs.append(f"methods pageerror: {e}"))
    mg.goto(URL.replace("index.html", "methods.html"), wait_until="networkidle")
    mg.wait_for_timeout(2500)
    # Open a box that has numbers on its face, so the pane is not the default one.
    mg.locator("#flow-svg g.node").nth(2).click()
    # The hover tooltip follows the click; park the pointer so it is not in the shot.
    mg.mouse.move(2, 2)
    mg.wait_for_timeout(400)
    flow = mg.locator("#flow").locator("xpath=following-sibling::div[@class='wrap'][1]")
    flow.scroll_into_view_if_needed()
    mg.wait_for_timeout(400)
    flow.screenshot(path=f"{P}-flow.png")
    fh = mg.evaluate("document.body.scrollHeight")
    b.close()

print(f"desktop {h} px (cut at {half}), phone {ph} px, methods {fh} px; "
      f"console {errs or 'clean'}")
