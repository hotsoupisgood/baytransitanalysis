# House style — lucashenry.xyz

Read-only survey of Puck's site, 2026-10-18, done to make this dashboard look
like it belongs there. Sources: `https://lucashenry.xyz/css/index.css`, the
projects index, and `/projects/{hse-trolleys,llm-chart-sentiment,cap-index,mosaicofgenes}/`.
Short excerpts only, quoted as evidence for the rules below. Everything here
is description, not instruction: it does not change any number on the site.

## 1. Typography

- One body stack, system sans, no web font:
  `font-family: -apple-system, system-ui, sans-serif;`
- Mono is used only for chrome (site title, nav labels) via a token:
  `--mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, …`
  and separately for `pre, code`:
  `font-family: Consolas, Menlo, Monaco, "Andale Mono WT", …`
- Body size is browser default (16px, never set). Line height is the one
  deliberate choice: `p, .tmpl-post li { line-height: 1.45; }`
- Sizes are relative and few: nav labels `0.6875rem` uppercase with
  `letter-spacing: 0.06em`; site title `1.375rem`; card titles `1.25em`;
  dates and excerpts `0.8125em`; tags `0.75em`; `pre` `14px/1.375`.
- Line length is capped twice: text column `--content-max: 66rem`, and
  **images and captions at 44rem**:
  `p img { display:block; max-width: min(100%, 44rem); margin: 0 auto; }`

## 2. Palette

Light only — the site declares no dark mode and no `prefers-color-scheme`
block. Tokens, verbatim:

```css
--lightgray: #e0e0e0;  --gray: #C0C0C0;  --darkgray: #333;
--navy: #17050F;       --blue: #082840;  --white: #fff;
```

- Background `--white`, text `--darkgray`, both set on `html, body`.
- Links are `--blue` (`a[href] { color: var(--blue) }`), visited `--navy`.
  No underline removal, no hover colour change on body links.
- Secondary text is a literal `#666` (nav, dates, card excerpts).
- Rules and borders are `--lightgray`; tag borders `--gray`.
- Code blocks: `background-color: #f6f6f6;` `padding: 1em`.
- The only saturated colour in the whole stylesheet is the warning box:
  `.warning { background-color: #ffc; padding: 1em 0.625em; }` — a muted
  cream, not red. This is the model for our EXPLORATORY banner.
- Accent-as-colour barely exists; emphasis is weight, rules and whitespace.

## 3. Figure conventions

- Markup is plain: `<p><img src="/img/…" alt="Annual cycle peak timing by
  region, weeks from New Year"></p>`. No `<figure>`, no `<figcaption>`.
  A `.caption` class exists (centred, 44rem) for when a caption is needed.
- **The caption is the prose.** The paragraph after the image names the
  figure and reads it: "In the plot above, every region has a sharp dip right
  around New Year, marked in light blue…", "Below is the modeled peak timing
  by region, with winter, December through February, shaded in blue."
- Alt text is a full descriptive sentence fragment with units.
- Uncertainty is written in words, inline, with the interval right after the
  point estimate: "8.13 trolleys per 10,000 population, more than three
  times HSE Dublin and North East's 2.27"; "bottoms out the week of August
  19th at 7.29 fewer trolleys per 10,000 than the regional level, with a 95%
  interval that excludes zero"; "its true mean ratio could plausibly sit
  anywhere from about 3.6 to 8.4".
- Negative results are stated as such, not hidden: "Mid West is the
  exception, where the timing is uncertain"; "isn't significantly different
  from eleven other countries, including first place Australia".
- He distinguishes what a number *is*: "That is a disagreement between two
  ASR systems, not a word error rate"; "which is self-consistency, not
  validity". Keep that habit on every panel.

## 4. TLDR-first structure

Every analysis post opens with a literal `TLDR:` label and 3–5 bullets, then
the prose. The bullets are first person, past tense, one claim each, numbers
with units, and the **last bullet links the interactive thing**:

> - We fit a Bayesian time series model to scraped Irish hospital trolley
>   counts, weekly for each of the six HSE regions from 2023 to 2026, scaled
>   per 10,000 population.
> - Found a west to east pattern. …
> - The interactive results are at [hsereport.ie](https://hsereport.ie/).

Body order after the TLDR: why the question matters → what the data is → how
it was built → each result with its figure → a sensitivity or audit → one
"a future direction I would like to try" paragraph → code/data links.

## 5. Interactive and external tools

- Presented in one plain sentence with an inline link, never a button or a
  card: "The interactive results are at [hsereport.ie](https://hsereport.ie/)",
  "[Here is a link to the website](https://mosaicofgenes.com/)", "All of this
  can be seen in the [dashboard](…)".
- Code always gets its own link in the closing line: "Code and data pipeline
  are on [GitHub](…)". `mosaicofgenes` shows the minimal form: three
  sentences, the live link, the stack ("I used Vue.js and Java
  (SpringBoot/Webflux)").

## 6. Eleventy front matter and tags

Eleventy v1.0.2. Posts live at `/projects/<slug>/`, dates render as
`15 Oct 2026`, and the listing uses `.cardlist-item` (title, date, excerpt,
thumb). Front matter shape:

```yaml
---
title: "Do LLMs Frame the Same Chart Differently?"
date: 2026-10-15
tags: ["llm", "statistics", "sentiment analysis"]
description: "…"   # one sentence, becomes the card excerpt
---
```

The complete tag vocabulary on the site today (use these, do not invent):
`android`, `app`, `bayesian`, `bioinformatics`, `bootstrap`, `crypto`,
`eleventy`, `flask`, `game`, `ios`, `jags`, `java`, `learning`, `llm`,
`node`, `python`, `sentiment-analysis`, `statistics`, `time-series`, `vue`,
`web-app`. For this project: `statistics`, `bayesian`, `llm`, `python`.

## 7. Voice

Short declaratives. First person, "we" when the work was shared, "I" for a
judgement call. Methods get a bolded run-in lead and a few sentences:
"**Power calculation.** Running LLMs on images is relatively expensive, so I
wanted to know how many replies I actually needed before spending anything."
Decisions are dated and the reasoning is shown, including the ones that went
badly: "I kept n at 250 … and froze that before collecting any more data."
Jargon is defined on first use in plain words. No exclamation marks, no
emoji, no hype adjectives.

**Superseded for the dashboard, 2026-10-21.** The owner's dashboard language
rules replace this section on the dashboard and the Methods page: formal
register, no first person, short descriptive sentences, precise claims that
name who was measured, what was measured and how much, uncommon terms defined
where they first appear, and no em dashes. The rules above still apply to
posts and to the rest of this file. The dashboard rules are in
`~/.openclaw/workspace/style/DASHBOARD-LANGUAGE.md`.

## 8. CSS variables for this dashboard

Drop-in token block for `site.css`, mapping his tokens onto the names the
panels already use so no panel markup has to change meaning:

```css
:root {
  /* his tokens, verbatim */
  --content-max: 66rem;
  --fig-max: 44rem;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas,
          "Liberation Mono", monospace;
  --code: Consolas, Menlo, Monaco, "Andale Mono WT", "Lucida Console",
          "DejaVu Sans Mono", "Liberation Mono", "Courier New", monospace;
  --lightgray: #e0e0e0;
  --gray: #C0C0C0;
  --darkgray: #333;
  --navy: #17050F;
  --blue: #082840;
  --white: #fff;
  --muted-ink: #666;
  --pre-bg: #f6f6f6;
  --warn-bg: #ffc;

  /* dashboard aliases */
  --bg: var(--white);
  --ink: var(--darkgray);
  --muted: var(--muted-ink);
  --panel: var(--white);
  --border: var(--lightgray);
  --grid: var(--lightgray);
  --accent: var(--blue);
  --link-visited: var(--navy);
  --pending: var(--gray);
}
```

Rules that follow from the survey: no dark-mode block, body
`font: 16px/1.45` system sans, images and captions capped at `--fig-max`,
figure captions as muted prose under the image, the EXPLORATORY banner
styled as `.warning` (`#ffc`, no border colour, no bold red), and the
category colours kept only where they carry data (map, pills, highlight
marks) — pulled toward his restraint everywhere else.

## TLDR rule (owner, 2026-10-18 21:16 PDT)

"tldr can be vague, and point toward the results like my other tldr.
details can be in methods." So: each panel/page TLDR is 3 to 5 short
first-person past-tense bullets that say what was looked at and gesture at
what came out ("the public leaned one way on housing and the other on
transit; the dais mostly stayed neutral"), one number at most per bullet,
no intervals, no instrument names, no caveats; the last bullet points to
the panel or the Methods page for the detail. Intervals, hashes, κ, model
diagnostics and caveats live in the panel body and on Methods.

Clarified 21:16: **no numbers in a TLDR at all**. Numbers appear only in
the results body of a panel and on Methods. ("like the only numbers need to
come in in the actual methods or for results stuff")

**Superseded for the dashboard, 2026-10-21 07:14.** Each panel opens with its
heading and a short descriptive sentence that states the measured quantity,
not with a bulleted TLDR of its own.

**Reinstated for the page, 2026-10-21 08:40.** "the tldr should replace the
first paragraph". The page itself opens with a TLDR block in the blog format:
3 or 4 bullets, first person, past tense, no numbers, with the last bullet
pointing at the plots and the Methods page. `ul.tldr::before` in site.css puts
the visible "TLDR:" label back. The per-panel TLDRs stay gone, so the page has
exactly one. The dashboard rules are in
`~/.openclaw/workspace/style/DASHBOARD-LANGUAGE.md`.

## Page order and no disclosures (owner, 2026-10-21 08:45 PDT)

"the site should begin after tldr with a sentence or two about the goal o fthe
classification ... then example of a nimby comment next to a yimby comment.
then get into the graphs. don't ahve all thoes drop downs that you have now.
just briefly summarize plots in short paragraphs."

The front page order is: h1, TLDR, the goal with the corpus scope and the one
exploratory sentence, the two links, why it matters, one NIMBY comment beside
one YIMBY comment, the plots, the footer. index.html carries **no `details`
element at all**. 49 of them came off the page on 2026-10-21:

- the episode reader and its two folds (the two example comments replace what
  it was there for; `data/episodes/*.json` is unchanged and nothing on the
  page fetches it now);
- the scatter's examples strip, 26 folds in all;
- "Show the numbers", one statistics table per panel, 4;
- "What is wrong with this result", 4, with their own nested tables;
- "More figures", 3;
- "Which speaker is 'the public', and how that was decided";
- "The numbers, and what is wrong with this" under the role block;
- the "Renter share" fold, which the panel came out of;
- the methods footer's "The corpus and every instrument version".

Nothing was deleted from the data. Every row is still in `data/panels.json`,
every figure is still in `data/panels/`, and the caveats, tables and
provenance are on the Methods page, which rule 8 of DASHBOARD-LANGUAGE.md
names as where they belong.

## Takeaway titles (owner, 2026-10-21 08:46 PDT)

"titles should be takeaways i forget if i put that there". Every heading under
the h1 is a sentence that states a finding, with its number where it has one.
The four panel headings are rendered from each panel's own first headline
sentence, so the title cannot drift from the data, and the summary paragraph
is the sentences after it rather than a restatement of the title (rule 14).
The goal, the links and the "why it matters" sentences carry no heading: a
label there would not be a takeaway, and a takeaway would restate the
sentence under it.

## Banner rule (owner, 2026-10-18 21:18 PDT)

"the yellow exploratory does not need to be there. just title the dashboard
something exploratory analysis." So: remove the cream EXPLORATORY banner and
the per-panel EXPLORATORY pills; put the word in the page title and h1
("an exploratory analysis of ..."); the limitations stay in the
Methods page and in each panel's results body, not as warning boxes.

Superseded in part, 2026-10-21 10:04: the h1 no longer carries the word, and
the exploratory sentence is stated once under the goal block instead (rule 12
of DASHBOARD-LANGUAGE.md). The title quoted above also carried the project's
internal working name, which is dropped from everything published; see
"Public name" below.

## Public name (owner, 2026-10-21 10:04 PDT)

"[the working name] is the internal name. drop it for the live site." Nothing
published carries the project's internal name: not the title tag, the h1, the
app bar, the footer, the Methods page, or text baked into a figure. Where a
name is needed, the site is "Bay Area Transit Analysis", which matches the
repository. Repo-relative file paths on Methods may still contain the
directory name: rule 10 makes provenance the Methods exception.
`copy_rules.json` carries a FAIL rule, `internal-name`, whose lookarounds
spare those paths.

## The first screen (owner, 2026-10-21 10:04 PDT)

"i want the images to start on the first page of the website. like when i load
it it should be there. so either remove some text or put it to the right of
the text." The TLDR, the goal block and a lead chart are one band: the chart
stands in a second column to the right of the text on a desktop, and comes
straight after the TLDR on a phone, with the goal, the links and the examples
below it. The chart is never shrunk to fit; the TLDR's four bullets were cut
to one sentence each instead. Measured on first load, with no scrolling: at
1280x800 the chart occupies 236 to 399 px, and at 390x844 it occupies 544 to
710 px, so it is whole on both, and whole at 390x720 as well.

The chart is the TLDR's third bullet drawn: two bars, members of the public
beside the officials, of the share of scored statements that took a side. Its
two shares are read back out of the `role_side` headline in `panels.json`
rather than typed into the page or added to the export, so the chart cannot
carry a number the export does not. The role block further down drops the
sentence that quoted those two shares, so each number is said once (rule 14).

## The map (owner, 2026-10-21 10:06 PDT)

"for the map. just use all the same shape. size based on n data. no need to
have the clickable panel on the right. its too much. keep info in the hover."
Every body is a circle whose area is the hours of that body's meetings that
were transcribed, which the legend says; the exported data carries no per-body
count of scored statements, only per-body episode counts, and no exporter was
re-run to add one. The square, triangle and diamond for council, transit board
and county are gone, and the body type is a word in the hover text. The detail
panel beside the map, its click handler, the `#body` deep link and the
scatter's body filter went with it, and so did the fetches of
`data/meetings.json` and `data/episodes.json`, which were 3.5 MB of every page
load. The hover text carries the body's name and type, its meetings and hours,
how much was transcribed, how many episodes passed the quality gate and were
scored, and the topic split, each with its unit. It opens on hover with a
mouse and on a tap on a touch screen.

## App bar and card (reference, 2026-10-21 09:54)

`style/DASHBOARD-REFERENCE.md` describes how the owner builds his own
dashboards. Applied here: one centred white card on a light grey page, prose
capped narrower than the card with figures running wider (rule 1); a thin
coloured app bar with the site name at the left and the views at the right,
the current one marked (rules 2 and 3); a one-sentence caption under each
chart (rule 5); one colour per series, defined once as a CSS variable read by
both the stylesheet and the chart code (rule 7); the value printed inside the
thing it describes rather than in a tile (rule 12); a closing sentence naming
the data, the sources and the code (rule 14); and a phone layout of one
column whose app bar wraps rather than scrolling off (rule 15).
