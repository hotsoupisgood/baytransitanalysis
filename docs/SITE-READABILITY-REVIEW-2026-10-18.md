# Readability review — 2026-10-18

Read-only review of the dashboard at `http://127.0.0.1:8766` (commit `8c89698`),
plus `flow.html` and `methods.html`. Measured in a real browser at 1440×900 and
390×844. The persona: someone who clicked a link from lucashenry.xyz, is not a
researcher, and will leave the moment it looks like homework.
Screenshots: `/tmp/thtrans-readability/`.

## 1. Verdict

I'd read the TLDR, click the map, and enjoy it — then quit about a screen and a
half into **Stance**, at the first table of log-odds, 90 % intervals and
`Pr(>0)`. On my phone I'd quit sooner: the page scrolls sideways and the map is
below the first screen.

## 2. Measurements

| Panel / page | Screens (1440×900) | Words | Tables (rows) | Figures |
|---|---|---|---|---|
| Head: title + sub + TLDR | 0.6 | 233 | 0 | 0 |
| Map + body detail | 0.7 | 67 (+447 on click) | 1 (29) | 1 |
| Episode reader | 0.6 | 417 (~170 are speech) | 1 | 0 |
| Stance: the role gap | 3.3 | 1,589 | 4 (34) | 6 |
| NIMBY / YIMBY | 3.9 | 1,932 | 4 (53) | 8 |
| Renter share | 4.5 | 1,086 | 1 (34) | 14 |
| Topic time per meeting-hour | 1.4 | 716 | 1 (6) | 0 |
| Models | 0.4 | 272 | 0 | 4 |
| Methods block (index) | 0.7 | 556 | 0 | 0 |
| **index.html total** | **16.2** | **6,589** | **10 (127)** | **~33** |
| index.html @ 390×844 | **32.6** | — | 9 of 11 overflow | — |
| methods.html | 10.3 | 3,756 | 4 | 1 |
| flow.html | 2.6 | 842 | 0 | 1 |

Time to a result: a visual at ~3 s (map top edge at 577 px, inside screen one);
a result *figure* needs 1.4 screens of scrolling. On mobile the map starts at
1,003 px — nothing but words on screen one.

Walls of numbers, in order hit: Stance's two stacked stats tables (tail shares,
then ordinal fits with `Pr(>0)` and a `.csv` filename); NIMBY's 13-row cluster
table plus a slash-notation footnote (`45% / 42% / 12% (Y/N/neutral)` ×4);
Renter share's raw 13-row disagree-rate table between two dense paragraphs
containing `SELF_ID`, backticked field names and `46.0 / 42.9 / 11.2 →
47.3 / 39.8 / 12.9`.

## 3. Changes that would help most, ranked

1. **Stop the sideways scroll on mobile.** Body is 613 px wide in a 390 px
   viewport; 9 of 11 tables overflow. Wrap every table in a scroll container
   (or collapse per #2–#4). Biggest single win — mobile is 32.6 screens.
2. **Renter share (1,086 words, 14 figures, 4.5 screens):** collapse the 13-row
   disagree-rate table and the flip table behind one sentence each and a
   "show the numbers" toggle. This is the worst wall on the page.
3. **One figure per panel above the fold, the rest behind "more figures".**
   Renter share has 14; Stance 6; NIMBY 8.
4. **NIMBY / YIMBY (1,932 words, 53 table rows):** collapse the cluster table
   behind "show the numbers", and cut the slash-notation footnote to one
   sentence — the route breakdown belongs on Methods.
5. **Stance (1,589 words, 4 tables):** keep the "In one line." lead, which
   works, and put both stats tables behind one toggle so the lead and the
   figure sit together.
6. **Move Renter share and Rates below a "more" fold**, after the three
   headline panels, so the page reads as four screens with an opt-in tail.
7. **Episode reader (417 words, ~170 of them actual speech):** lead with the
   quoted turn. Move the five metadata lines — QA OK, `db`, `pipeline/roles.py`,
   k=3 votes, the ± spread — behind "about this excerpt".
8. **Cut the 48-word subtitle to two lines** (it restates TLDR bullets 1–2),
   and give **methods.html** (3,756 words, 10.3 screens) a TLDR; it is the only
   page without one.

## 4. What already works

- **The TLDR.** Five bullets, 139 words, no numbers, plain past tense. You know
  the three findings before you scroll once. It matches the house rule exactly.
- **The map.** 33 clickable marks, and the click pays off: Hayward returns a
  readable plain-language summary ("29 regular meetings (68.5 h)… 37 % density
  / 63 % transit") before any table. Best moment on the site.
- **The reader's quoted speech**, with density and transit keywords highlighted
  in the turn. It is the only place a general reader hears an actual person at
  an actual meeting, and it is genuinely interesting.
- Bonus: `flow.html` at 2.6 screens and one diagram is the right length.

## 5. Suggested reading order

1. Title + TLDR (unchanged)
2. Map + body detail (unchanged)
3. Episode reader — ideally with one episode already loaded, so the payoff
   arrives without a click
4. **Topic time per meeting-hour** — currently 4th. It is the shortest panel
   (716 words, 1.4 screens), the most concrete, and it is the TLDR's first
   finding. Promote it.
5. **NIMBY / YIMBY** — the label a general reader already knows
6. **Stance: the role gap**
7. — "more" fold —
8. Renter share, Models
9. Methods and flowchart links

---

## 6. After the changes — re-measured 2026-10-18 22:2x

Same browser, same two widths, same page (`http://127.0.0.1:8766`), commits
`04d74c6` (pass 1) and `36294fc` (pass 2). "Visible by default" means what is
on screen before any disclosure is opened; a closed `<details>` counts only its
summary line. **No number, figure, table or sentence was removed** — every one
of them is still on the page, one click away.

### Headline

| Measure (1440×900) | Before | After | Target |
|---|---|---|---|
| Screens, whole page | 16.2 | **5.2** | — |
| Screens before the "more" fold | 16.2 | **4.9** | ≤ 7 |
| Words visible by default, above the fold | 6,589 | **1,764** | < 2,200 |
| Tables visible by default | 10 (127 rows) | **0** | 0 |
| Horizontal overflow | 0 px | **0 px** | 0 |
| Map top edge | 577 px | **505 px** | screen 1 |
| First result figure | 3,074 px (1.4 screens) | **2,952 px** | — |

| Measure (390×844) | Before | After | Target |
|---|---|---|---|
| Screens, whole page | 29.0 | **8.5** | — |
| Horizontal overflow | **223 px** | **0 px** | 0 |
| Tables overflowing | 9 of 11 | **0** | 0 |
| Map top edge | 1,003 px (screen 2) | **734 px** (screen 1) | screen 1 |

At 390 px with *every* disclosure on the page forced open, horizontal overflow
is still 0 px (45.4 screens). Console is clean on `index.html`, `methods.html`
and `flow.html`.

### Per panel, visible by default at 1440×900

| Panel | Words before → after | Tables before → after | Height after |
|---|---|---|---|
| Head: title + sub + TLDR | 233 → 180 | 0 → 0 | 0.3 screens |
| Map + body detail | 67 → 67 | 0 → 0 | 0.7 |
| Episode reader | 417 → 305 | 1 → 0 | 0.6 |
| Topic time per meeting-hour | 716 → 239 | 1 → 0 | 0.6 |
| NIMBY / YIMBY | 1,932 → 384 | 4 → 0 | 1.0 |
| Stance: the role gap | 1,589 → 500 | 4 → 0 | 1.1 |
| Renter share | 1,086 → 0 (in the fold) | 1 → 0 | — |
| Models | 272 → 0 (in the fold) | 0 → 0 | — |
| Methods block | 556 → 53 | 0 → 0 | 0.1 |

### What moved behind a disclosure

Per panel, in document order: the lead figure stays visible, then
*Show the numbers — &lt;table title&gt;* (one per statistics table, each preceded
by one plain sentence that says what the table holds), *More figures (n)*,
*Which speaker is "the public" here*, and *What is wrong with this result —
n points, in full* (the caveat prose and any table under it). The reader keeps
the quoted turn and hides *About this excerpt* (the five metadata lines) and
*Read the whole episode, turn by turn*. The methods footer keeps its opening
paragraph and folds *The corpus and every instrument version, as exported*.
Renter share and Models sit whole inside the page-level "more" fold.

### Still open

- **Topic time has no figure of its own**, so the "one figure per panel above
  the fold" rule has nothing to apply there; `panels.json` ships `figs: []` for
  it. Making one would be a figure change, which this pass does not do.
- Panel figure titles are still derived from the filename
  (`public yimby nimby classifier stars`), which reads like a filename.
- The phone head is 734 px before the map: title, two-line subtitle and the
  five TLDR bullets. It fits, but only just; shortening it further would mean
  cutting TLDR text, which the house rule fixes at three to five bullets.
- `methods.html` is still 10.3 screens (15.5 on the phone). It has a TLDR and
  no overflow, and it is the page where the detail is supposed to live, so it
  was left long.

---

## 7. Second pass — confirmation, 2026-10-18 ~22:4x (commit `40c6035`)

Same persona, same browser, same two widths. Read-only. Screenshots:
`/tmp/thtrans-readability2/`. The targets in §6 were **not** assumed; every
number below was re-measured on the live page after a fresh load.

### Verdict

I'd now read the TLDR, click the map, read the episode, get through Topic time,
and reach the new scatter — which is the best thing on the site, better than the
map. I don't quit at a stats table any more; **there is no table on screen by
default anywhere on the page.**

I stop twice, both inside NIMBY / YIMBY, and both are new since §6:

1. The paragraph beginning *"A derived view, no new scoring: the unit is one
   anonymous speaker cluster inside one episode (never linked across meetings,
   plan/DECISIONS.md #15)…"* — 103 words of construct-definition sitting
   **above** that panel's TLDR bullets. It is the first jargon wall on the page
   and it is in the position that should be the hook.
2. The examples strip renders **six quote cards open by default** on desktop
   (521 words of block quote). On the phone the same 26 cards are all closed.
   Desktop regressed; mobile did not.

### Measurements: before → after (§6) → now

| Measure (1440×900) | Before | After | **Now** |
|---|---|---|---|
| Screens, whole page | 16.2 | 5.2 | **6.92** |
| Words visible by default | 6,589 | 1,764 | **3,941** |
| Tables visible by default | 10 (127 rows) | 0 | **0** ✅ |
| Horizontal overflow | 0 px | 0 px | **0 px** ✅ |
| Map top edge | 577 px | 505 px | **505 px** ✅ |
| Screens, every disclosure open | — | 45.4 | **23.7** |

| Measure (390×844) | Before | After | **Now** |
|---|---|---|---|
| Screens, whole page | 29.0 | 8.5 | **9.52** |
| Horizontal overflow | 223 px | 0 px | **0 px** ✅ |
| Tables overflowing | 9 of 11 | 0 | **0** ✅ |
| Map top edge | 1,003 px | 734 px | **734 px** ✅ |

Per panel, visible by default at 1440×900:

| Panel | Words after → now | Screens now |
|---|---|---|
| Head + map | 180 → 259 | 1.17 |
| Episode reader | 305 → 332 | 0.74 |
| Topic time | 239 → 239 | 0.65 |
| **NIMBY / YIMBY** | **384 → 1,129** | **2.75** |
| Stance: the role gap | 500 → 513 | 1.26 |
| Methods block | 53 → 78 | 0.25 |

The whole 1,742-word / 1.72-screen delta against §6 is NIMBY / YIMBY. Every
other panel held. No broken images (32 checked), no table overflow at either
width even with all 52 disclosures forced open, console clean.

### What still hurts, ranked

1. **The six open example cards (desktop only).** 521 words of quote on screen
   before anyone asks. The phone already closes all 26; make desktop match —
   this one attribute change recovers most of the regression.
2. **The "A derived view, no new scoring…" paragraph** sits above the NIMBY /
   YIMBY TLDR. Move it below the bullets or behind *What is wrong with this
   result*. `plan/DECISIONS.md #15` and `≥ +0.5` are not hook material.
3. **Figure titles still read like filenames** (known gap, unchanged): twelve of
   them, including `public yimby nimby swagit transcript stars`,
   `public vs dais agenda index`, and `cue public self dais`. The last is
   unreadable to anyone who has not read Methods.
4. **Topic time still has no figure of its own** (known gap, unchanged). It is
   the TLDR's first finding and the shortest panel at 0.65 screens — it is the
   cheapest place on the site to add a payoff, and the only panel with none.
5. **Phone: the scatter dots are not tappable.** Rendered radius is 2.55–5 px
   inside a 328 px SVG, and the caption promises "or tap one". Axis labels
   render at 10 px pre-scale, so "hostility of delivery" is illegible.
6. **Caption tail noise.** The lead figure caption ends with the raw repo path
   `data/panels/fig_public_yimby_nimby_classifier_stars.png`, and the panel lead
   carries a `corrected 2026-10-18 after role audit` badge. Both are notes to
   self, in the reader's first two sentences.

### The new scatter and examples strip

The scatter itself is the strongest addition to the site. Hover fills a card
with the real quoted speech, the route selector reprices the chart honestly
(classifier 198 → agenda_index 49 speakers in 4 bodies), click-to-pin holds
through subsequent hovers, a second click releases it, and the examples strip
re-keys to the selected route. The legend, neutral band and both axis arrows
are plain English. Nothing errored.

Three things confuse:

- **The card goes stale on route change.** Switching classifier → agenda_index
  redraws 198 dots to 49, but the card below keeps showing the previously
  hovered classifier speaker, who may not exist on the new route. Clear the
  card (or re-pick) on `change`.
- **"Anonymous" is contradicted by the cards.** The panel says "one anonymous
  speaker cluster … never linked across meetings", and the first card I opened
  begins *"Mayor and council members, I'm …"* with a full self-identified name.
  Both are defensible — it is public record, and the pipeline genuinely does not
  link clusters — but a general reader reads it as a contradiction. One clause
  ("speakers are not tracked across meetings, though many name themselves")
  fixes it.
- **The 26 cards are `<details>` that are already open**, so the twisty is
  decoration on desktop and a real control on the phone. Same markup, two
  different contracts.

### Ready to show a friend?

**Yes** — with item 1 done first. It is a one-attribute change and it takes the
page back under 2,000 visible words and ~5.2 screens. Items 2 and 6 are ten
minutes of copy. Items 3, 4 and 5 can wait; none of them is why someone leaves.

---

## 8. Third pass — tooltips and the six items, 2026-10-18 ~23:0x

Same browser, same two widths, same page. Every number below was re-measured on
a fresh load after the changes, not assumed. Screenshots: `/tmp/thtrans-final/`.

### The dot tooltip (owner, 22:37: "could we make tooltips on the dots with text?")

The hover card under the scatter stays — it is where a thumb can read a long
quote without covering it — but a dot now also carries a real tooltip, verified
in the browser:

- **Opens on hover, no click, no delay.** Measured 2.8 ms from `pointerenter`
  to a visible tooltip; there is no timer in the path.
- **Follows the pointer** while hovering, flips to the other side of the dot
  near a viewport edge, and is then clamped inside the viewport.
- **Quote first**, then body · date · label, then sub-topic, the two means and
  the *watch at* link — the same fields the card below carries.
- **Closes on mouse-out**, unless pinned.
- **Click pins it**: the tooltip re-anchors to its dot, becomes clickable (so
  the *watch* link is reachable), and holds through later hovers. A second
  click on the same dot releases it. It tracks its dot on scroll and re-anchors
  on resize.
- **Touch: one tap shows, a second tap on the same dot pins, a third releases.**
  The synthetic click that follows a tap is swallowed, so a first tap can never
  pin. Verified by dispatching `pointerType: "touch"` at 390 px.
- A route change clears hover, pin and card together (item 6 below).

The native SVG `<title>` on each dot was removed — it was a second, slower
tooltip saying less.

### The six items

1. **Example quote cards closed on desktop too.** `.qex[open]` is 0 at both
   widths. This is the whole of the §7 regression and it is gone.
2. **The "derived view" paragraph** is two sentences and renders *below* the
   panel's bullets (the `head()` order is now TLDR → bullets → lede, on every
   panel). The thresholds, the transit exclusion and the no-pooling rule moved
   into a new first caveat, *How a speaker gets a label, and what the label is
   not* — nothing was deleted.
3. **Written titles and captions for every figure.** All 33 figure headings on
   the page are English sentences from `panels.json`; none is a filename. The
   raw `data/panels/…` path tail is gone from every caption (0 `<code>` tails
   in `.fig p`), as is the `corrected 2026-10-18` badge beside the panel TLDRs.
   The correction itself is written out in the caveats, where the numbers it
   moved are. The renter panel's 14 figures had no captions at all before.
4. **Topic time has its own figure.** Horizontal transit/density bars with 95 %
   intervals, one pair per complete-coverage body and route, drawn client-side
   with D3 from `rates.complete` — the rows the table under it prints. No new
   number. The caption reads the chart in house style.
5. **Phone scatter.** The viewBox is now the rendered width, so one unit is one
   CSS pixel: dot radius **6–11 px** (was 2.55–5) and axis labels **12 px**
   (was 10 px pre-scale, ~5 px rendered). "Tap a dot" is true now.
6. **The card clears on route change** — hover, pin, tooltip and card.
7. **"Anonymous" resolved in one clause**: the lede says clusters are not
   linked across meetings, so no one is followed from one meeting to the next,
   and what a speaker says — including any name they give on the record — is
   shown as spoken. The word *anonymous* is gone from the panel's bullets.

The NIMBY/YIMBY panel's lead figure is now the scatter: the four route bar
charts moved together into one disclosure. They are the same shares as the
table beneath them, and one of them above the scatter cost half a screen and a
caption that had to say "the bars above".

### Measurements: §6 → §7 → now

| Measure (1440×900) | §6 | §7 | **Now** | Target |
|---|---|---|---|---|
| Screens, whole page | 5.2 | 6.92 | **6.43** | ~5.5 |
| Words visible by default | 1,764 | 3,941 | **2,061** | < 2,000 |
| Tables visible by default | 0 | 0 | **0** ✅ | 0 |
| Horizontal overflow | 0 px | 0 px | **0 px** ✅ | 0 |
| Map top edge | 505 px | 505 px | **505 px** ✅ | screen 1 |
| Screens, every disclosure open | 45.4 | 23.7 | **26.0** | — |

| Measure (390×844) | §6 | §7 | **Now** | Target |
|---|---|---|---|---|
| Screens, whole page | 8.5 | 9.52 | **9.47** | — |
| Words visible by default | — | — | **2,051** | — |
| Horizontal overflow | 0 px | 0 px | **0 px** ✅ | 0 |
| Tables overflowing the page | 0 | 0 | **0** ✅ | 0 |
| Map top edge | 734 px | 734 px | **734 px** ✅ | screen 1 |
| Scatter dot radius | — | 2.55–5 px | **6–11 px** ✅ | ≥ 6 px |
| Scatter axis labels | — | ~5 px | **12 px** ✅ | 12 px |
| Example cards open by default | 0 | 0 | **0** ✅ | 0 |

Per panel, visible by default at 1440×900:

| Panel | Words §7 → now | Screens now |
|---|---|---|
| Head + map | 259 → 206 | 0.97 |
| Episode reader | 332 → 305 | 0.61 |
| **Topic time** | 239 → **340** | **0.99** |
| **NIMBY / YIMBY** | 1,129 → **582** | **1.86** |
| Stance: the role gap | 513 → 411 | 1.03 |
| Methods block | 78 → 53 | 0.13 |

32 images, none broken. 11 tables, all behind a disclosure; with all 52
disclosures forced open, no table overflows the page at either width (the wide
renter correlations table scrolls inside its own `.tscroll` box, as before). Console clean on `index.html`, `methods.html` and `flow.html`.
`analysis/test_pipeline.py` section 33: 50 checks, 0 FAIL.

### Against the targets

Words are **2,061 against a target of under 2,000** — 61 over, and I stopped
cutting there. Everything still on screen is a result, a figure caption or a
sentence standing in for a folded table; the next 61 words would have to come
out of the panel bullets, which are the findings themselves and which the house
rule fixes at three to five.

Screens are **6.43 against ~5.5**, and the gap is the two things this pass was
asked to add: Topic time's new figure (+0.34 screens, the panel went 0.65 →
0.99) and the scatter's phone-legible frame. §6's 5.2 screens had neither a
scatter nor a figure in Topic time. Dropping back to 5.5 now means removing a
figure, which is the opposite of item 4.

### Still open

- `methods.html` is still 10.3 screens; it is the page where detail lives.
- Route names (`local_asr`, `agenda_index`, `swagit_transcript`) appear in
  figure titles with their underscores. They are the pipeline's own identifiers
  and are used in the same form everywhere else on the site, so they were left
  as they are rather than prettified in one place only.
