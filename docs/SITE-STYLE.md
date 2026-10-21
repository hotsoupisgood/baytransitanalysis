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

## Banner rule (owner, 2026-10-18 21:18 PDT)

"the yellow exploratory does not need to be there. just title the dashboard
something exploratory analysis." So: remove the cream EXPLORATORY banner and
the per-panel EXPLORATORY pills; put the word in the page title and h1
("thtrans — an exploratory analysis of …"); the limitations stay in the
Methods page and in each panel's results body, not as warning boxes.
