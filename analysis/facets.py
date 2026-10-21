#!/usr/bin/env python3
"""
facets.py -- the two instrument facets of the meeting corpus, and the guard that
stops a figure pooling across them by accident.

WHY THIS EXISTS. Two columns of `episodes` are not corpus variables, they are
INSTRUMENT ids. Pooling over either one makes a figure that averages two
different measuring devices:

  method       the LOCATE route. 'classifier' episodes were found by a Claude
               pass over caption windows; 'agenda_index' episodes were read off
               English agenda item titles; 'local_asr' would be a Mac ASR track;
               'swagit_transcript' is Swagit's own voice-to-text transcript.
               Their recall differs and is not known to be equal, so
               plan/T2-PLAN.md 3 and 336-340 say to report per method, never
               pooled. A per-meeting-hour rate mixing routes is a rate of
               nothing in particular.

  role_source  where dais/public came from. 'db' roles were written into
               state.sqlite speaker_roles by the labeller at ingest; 'computed'
               roles are recomputed at load time from cue phrases because the
               episode predates the labeller (batch b13). Every role contrast --
               which is most of A0 and all of A1 -- is therefore measured with a
               different instrument in the two halves.

               On the T1 pilot the two differ by ~44x in unknown-role share
               (6.6 % of computed-source words against 0.15 % of db-source), and
               A0-RESULTS 4 read that as an instrument difference. On the wide
               corpus it is 6.63 % against 6.56 % -- indistinguishable. What
               actually varies is the BODY: the db labeller leaves 0.0 % unknown
               at BART, Oakland, Hayward and SamTrans and 71 % at Fremont. So
               facet by role_source because the instruments are not known to be
               equal, not because this particular gap is established.

HOW IT IS ENFORCED. facet_sweep.py writes one corpus slice per facet level and
stamps it with an `a0_facet` table. a0_plots.py, a0_terms.py, a0_bridge.py and
a1_sentiment.py call `require_single(con)` before drawing anything: a corpus
holding more than one level of a facet, with no stamp, is refused with the
command to run instead. A genuinely pooled figure is still available -- `--pool`
-- but it has to be typed, and everything it writes is stamped POOLED in the
footnote and the filename, so it cannot be mistaken for a per-facet result.

Nothing here filters text or changes a count: a facet slice is the same
tokeniser and the same lexicon over a subset of episodes.
"""
import sqlite3

# facet name -> (episodes column, human label, whether it splits the exposure
# denominator's numerator in meeting_exposure)
FACETS = {
    "method":      ("method", "locate route", True),
    "role_source": ("roles_source", "role instrument", False),
}

LABELS = {
    "classifier":   "classifier (Claude over caption windows)",
    "agenda_index": "agenda index (English agenda titles)",
    "local_asr":    "local ASR (Mac track)",
    # 2026-09-24: Swagit's own voice-to-text transcript of a Petaluma council
    # meeting, read by the same stage 1 and codebook as a caption track but
    # produced by a different transcriber (pipeline/swagit.py).
    "swagit_transcript": "Swagit portal transcript (voice-to-text)",
    "unknown":      "route not recorded",
    "db":           "roles from speaker_roles (labeller at ingest)",
    "computed":     "roles recomputed from cue phrases at load time",
    "json":         "roles carried in the transcript JSON",
}

STAMP = "a0_facet"


def column(facet):
    try:
        return FACETS[facet][0]
    except KeyError:
        raise SystemExit(f"unknown facet {facet!r}; known: {', '.join(FACETS)}")


def levels(con, facet):
    """Levels of `facet` present in this corpus, most episodes first."""
    col = column(facet)
    return [r[0] for r in con.execute(
        f"SELECT {col}, COUNT(*) n FROM episodes GROUP BY 1 ORDER BY n DESC, 1")]


def present(con):
    """{facet: [levels]} for every facet with at least one level."""
    return {f: levels(con, f) for f in FACETS}


def stamp(con, facet, level):
    """Mark a sliced corpus so the figure scripts know it is already a facet."""
    con.execute(f"DROP TABLE IF EXISTS {STAMP}")
    con.execute(f"CREATE TABLE {STAMP} (facet TEXT, level TEXT)")
    con.execute(f"INSERT INTO {STAMP} VALUES (?,?)", (facet, level))


def stamped(con):
    """(facet, level) if this corpus is a facet slice, else None."""
    try:
        row = con.execute(f"SELECT facet, level FROM {STAMP}").fetchone()
    except sqlite3.Error:
        return None
    return (row[0], row[1]) if row else None


def describe(con):
    """One line naming the scope, for a figure footnote."""
    s = stamped(con)
    if s:
        return f"{FACETS[s[0]][1]} = {LABELS.get(s[1], s[1])}"
    got = {f: ls for f, ls in present(con).items() if len(ls) > 1}
    return ("POOLED over " + "; ".join(f"{FACETS[f][1]} ({', '.join(ls)})" for f, ls in got.items())
            if got else "single instrument")


def require_single(con, pool=False, script="analysis/a0_plots.py"):
    """Refuse to draw a figure that pools two instruments unless asked to.

    Returns the footnote fragment describing the scope. Raises SystemExit with
    the command to run instead when the corpus spans a facet and --pool was not
    given -- 'the figure is impossible to produce by accident' means exactly
    this: you get an error and a suggestion, not a plausible-looking plot."""
    s = stamped(con)
    if s:
        return describe(con)
    mixed = {f: ls for f, ls in present(con).items() if len(ls) > 1}
    if not mixed:
        return describe(con)
    if pool:
        return describe(con)
    names = ", ".join(f"{f} ({'/'.join(ls)})" for f, ls in mixed.items())
    raise SystemExit(
        f"refusing to draw a pooled figure: this corpus spans {names}.\n"
        f"  These are instruments, not variables (analysis/facets.py, plan/T2-PLAN.md).\n"
        f"  Draw it per facet:   .venv/bin/python analysis/facet_sweep.py --db <db>\n"
        f"  Or pool on purpose:  .venv/bin/python {script} --db <db> --pool\n"
        f"    (--pool stamps every figure and filename POOLED)")


def slice_db(src, dst, facet, level):
    """Copy `src` to `dst` keeping only the episodes at this facet level.

    meetings keeps every classified meeting -- the exposure DENOMINATOR is the
    meeting-hours the route was applied to, and every route was applied to every
    meeting in scope, so it does not split. The NUMERATOR columns are rewritten
    from meeting_exposure for the method facet; for role_source, which is a
    property of a transcript and not of locating, they are left pooled and the
    figure footnote says so."""
    import shutil
    shutil.copy(src, dst)
    col = column(facet)
    con = sqlite3.connect(dst)
    con.execute(f"DELETE FROM episode_segments WHERE episode_id NOT IN "
                f"(SELECT episode_id FROM episodes WHERE {col} IS ?)", (level,))
    con.execute(f"DELETE FROM episode_speakers WHERE episode_id NOT IN "
                f"(SELECT episode_id FROM episodes WHERE {col} IS ?)", (level,))
    con.execute(f"DELETE FROM episodes WHERE {col} IS NOT ?", (level,))
    if FACETS[facet][2]:
        con.execute("""UPDATE meetings SET
                         n_about = COALESCE((SELECT n_about FROM meeting_exposure x
                                             WHERE x.video_id = meetings.video_id AND x.method = ?), 0),
                         n_mention = COALESCE((SELECT n_mention FROM meeting_exposure x
                                             WHERE x.video_id = meetings.video_id AND x.method = ?), 0),
                         n_done = COALESCE((SELECT n_done FROM meeting_exposure x
                                             WHERE x.video_id = meetings.video_id AND x.method = ?), 0),
                         about_core_s = COALESCE((SELECT about_core_s FROM meeting_exposure x
                                             WHERE x.video_id = meetings.video_id AND x.method = ?), 0.0),
                         density_core_s = COALESCE((SELECT density_core_s FROM meeting_exposure x
                                             WHERE x.video_id = meetings.video_id AND x.method = ?), 0.0),
                         transit_core_s = COALESCE((SELECT transit_core_s FROM meeting_exposure x
                                             WHERE x.video_id = meetings.video_id AND x.method = ?), 0.0),
                         both_core_s = COALESCE((SELECT both_core_s FROM meeting_exposure x
                                             WHERE x.video_id = meetings.video_id AND x.method = ?), 0.0)""",
                    (level,) * 7)
    stamp(con, facet, level)
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    w = con.execute("SELECT COALESCE(SUM(words), 0) FROM episodes").fetchone()[0]
    con.close()
    return n, w
