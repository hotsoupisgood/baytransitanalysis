#!/usr/bin/env python3
"""
scripts/cluster_pilot.py -- unsupervised clustering pilot of density/transit speaker
turns (plan/CLUSTER-PILOT.md). EXPLORATORY; nothing here is an instrument result.

Question: without labels, do speaker turns cluster into anything like NIMBY /
YIMBY stances, or into something else (role, body, register)?

Unit and frame: the sibling NIMBY pilot's turn (scripts/nimby_pilot.py via
stance_s1_sample.corpus_turns): maximal run of one diarized cluster inside one
episode, redacted text (output/redacted/*.json, names -> [name]), >= 25 words,
family density | both, role dais | public. Anonymous per-episode clusters only
(plan/DECISIONS.md #15); nothing links a speaker across meetings.

Facets (analysis/facets.py never-pool rule): every clustering and every figure is
drawn inside ONE locate route (episodes.method) and ONE role instrument
(roles_source = 'db'; the 148 'computed' turns are left out rather than pooled).

Views:   lex    TF-IDF unigrams+bigrams (stopwords, fillers, [name] removed), LSA 100-d, l2
         style  22 register features (length, hedges, pronouns, questions, modals,
                VADER valence, plan/hostility.txt rate, ...), standardised
         embed  SKIPPED: no sentence-embedding model installed in the venv ($0, offline)
         nmf    NMF topics k = 6..12 on the TF-IDF matrix, turn -> argmax topic
Clusterers: k-means k = 2..8 (10 restarts), HDBSCAN (sklearn) on lex and style.
Model choice: silhouette and bootstrap stability (mean ARI of 10 subsample
refits against the full fit, over the shared points).
Validation (external only, never training): the sibling's Sonnet labels in
output/nimby_pilot/llm_labels.jsonl when present (ARI, NMI, purity); role; body.

    cluster_pilot.py run [--facets classifier,local_asr,...] [--out output/cluster_pilot]
"""
import argparse, csv, json, math, re, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline")); sys.path.insert(0, str(ROOT / "analysis")); sys.path.insert(0, str(ROOT / "scripts"))
import stance  # noqa: E402
import a0_terms as at  # noqa: E402
import a0_plots as ap  # noqa: E402
import word_freq as wf  # noqa: E402
import stance_s1_sample as s1  # noqa: E402

OUT = ROOT / "output" / "cluster_pilot"
NIMBY = ROOT / "output" / "nimby_pilot"
SEED = 20261018
MIN_WORDS = 25
KS = range(2, 9)
NMF_KS = range(6, 13)
N_BOOT = 10
MIN_FACET_TURNS = 150
LLM_TO_3 = {"YIMBY": "YIMBY", "NIMBY": "NIMBY", "neither": "neither", "not_about_housing": "neither"}
WORD_RE = stance.WORD_RE
PLACEHOLDER_RE = re.compile(r"\[[a-z_ ]+\]", re.I)
STOP = wf.STOP | wf.FILLERS | {"name", "uh", "um"}

HEDGES = ["i think", "i believe", "i guess", "i feel", "maybe", "perhaps", "sort of", "kind of", "probably",
          "possibly", "it seems", "seems like", "somewhat", "a little", "a bit", "not sure", "i suppose", "arguably"]
MODALS = ["should", "must", "need to", "needs to", "have to", "has to", "ought to", "could", "would", "might", "may", "shall"]
WE = ["we", "we're", "we've", "we'll", "we'd", "our", "ours", "us", "ourselves"]
I_ = ["i", "i'm", "i've", "i'll", "i'd", "my", "mine", "me", "myself"]
YOU = ["you", "you're", "you've", "you'll", "your", "yours"]
THEY = ["they", "they're", "they've", "their", "them"]
NEG = ["no", "not", "never", "nothing", "nobody", "can't", "cannot", "don't", "doesn't", "didn't", "won't", "isn't", "aren't", "wasn't"]
PROCEDURE = ["motion", "second", "agenda", "item", "staff report", "public comment", "roll call", "minutes", "consent",
             "resolution", "ordinance", "recommendation", "vote", "approve", "adopt", "continue", "hearing"]
NUMBER_RE = re.compile(r"\b\d[\d,\.]*\b|\b(percent|million|billion|thousand|dollars?)\b", re.I)


def log(msg, fh=None):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n"); fh.flush()


# ---------------------------------------------------------------- turns -------------
def load_turns():
    turns, missing, _p, n_eps = s1.corpus_turns(min_words=MIN_WORDS)
    out = [t for t in turns if t["family"] in ("density", "both") and t["role"] in ("dais", "public")]
    return out, missing, n_eps


def load_llm_labels():
    p = NIMBY / "llm_labels.jsonl"
    if not p.exists():
        return {}
    lab = {}
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("label") in LLM_TO_3:
            lab[r["turn_id"]] = {"raw": r["label"], "label3": LLM_TO_3[r["label"]]}
    return lab


# ---------------------------------------------------------------- views -------------
def clean_text(text):
    return PLACEHOLDER_RE.sub(" ", text)


def tokens(text):
    return [w for w in WORD_RE.findall(clean_text(text).lower()) if w not in STOP and len(w) > 2]


def analyzer(text):
    """unigrams + bigrams from the stop/filler-filtered token stream (bigrams across a removed stopword are allowed)."""
    w = tokens(text)
    return w + [f"{a} {b}" for a, b in zip(w, w[1:])]


def tfidf_view(texts, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(analyzer=analyzer, min_df=5, max_df=0.5, sublinear_tf=True)
    X = vec.fit_transform(texts)
    n_comp = min(100, X.shape[1] - 1, X.shape[0] - 1)
    Z = TruncatedSVD(n_components=n_comp, random_state=seed).fit_transform(X)
    return X, normalize(Z), vec


def count_phrases(text_l, phrases):
    return sum(len(re.findall(rf"(?<!\w){re.escape(p)}(?!\w)", text_l)) for p in phrases)


def hostility_pats():
    hostility = at.load_frames(ROOT / "plan" / "hostility.txt")
    return {k: [at.term_pattern(t) for t in ts] for k, ts in hostility.items() if not k.startswith("ambiguous_")}


STYLE_COLS = ["log_words", "wpm", "hedge_r", "we_r", "i_r", "you_r", "they_r", "we_minus_i", "question_r", "exclaim_r",
              "modal_r", "neg_r", "procedure_r", "number_r", "mean_sent_len", "ttr", "vader_compound", "vader_pos",
              "vader_neg", "hostility_r", "long_word_r", "frames_n"]


def style_view(turns, va, hpats, fpats):
    rows = []
    for t in turns:
        text = clean_text(t["text"]); tl = text.lower()
        words = WORD_RE.findall(tl); n = max(len(words), 1)
        sents = max(len(re.findall(r"[.!?]+", text)), 1)
        dur = max(t["end_s"] - t["start_s"], 1.0)
        v = va.polarity_scores(text)
        rows.append([
            math.log(n), n / dur * 60.0,
            count_phrases(tl, HEDGES) / n * 100, count_phrases(tl, WE) / n * 100, count_phrases(tl, I_) / n * 100,
            count_phrases(tl, YOU) / n * 100, count_phrases(tl, THEY) / n * 100,
            (count_phrases(tl, WE) - count_phrases(tl, I_)) / n * 100,
            text.count("?") / sents * 100, text.count("!") / sents * 100,
            count_phrases(tl, MODALS) / n * 100, count_phrases(tl, NEG) / n * 100, count_phrases(tl, PROCEDURE) / n * 100,
            len(NUMBER_RE.findall(text)) / n * 100, n / sents, len(set(words)) / n,
            v["compound"], v["pos"], v["neg"],
            sum(len(p.findall(text)) for ps in hpats.values() for p in ps) / n * 100,
            sum(1 for w in words if len(w) >= 9) / n * 100,
            sum(1 for ps in fpats.values() if any(p.search(text) for p in ps)),
        ])
    return np.array(rows, dtype=float)


# ---------------------------------------------------------------- clustering --------
def kmeans_fit(Z, k, seed):
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)


def silhouette(Z, lab):
    from sklearn.metrics import silhouette_score
    if len(set(lab)) < 2:
        return float("nan")
    return float(silhouette_score(Z, lab, sample_size=min(len(lab), 2000), random_state=SEED))


def stability(Z, k, full, seed, fit=kmeans_fit):
    from sklearn.metrics import adjusted_rand_score
    rng = np.random.default_rng(seed)
    n = Z.shape[0]; aris = []
    for b in range(N_BOOT):
        idx = rng.choice(n, size=int(0.8 * n), replace=False)
        lab = fit(Z[idx], k, seed + b + 1)
        aris.append(adjusted_rand_score(full[idx], lab))
    return float(np.mean(aris)), float(np.std(aris))


def sweep_kmeans(Z, seed):
    rows, labels = [], {}
    for k in KS:
        lab = kmeans_fit(Z, k, seed)
        sil = silhouette(Z, lab)
        st, sd = stability(Z, k, lab, seed)
        rows.append({"k": k, "silhouette": round(sil, 4), "stability_ari": round(st, 4), "stability_sd": round(sd, 4),
                     "sizes": "/".join(str(c) for _, c in sorted(Counter(lab).items()))})
        labels[k] = lab
    return rows, labels


def pick_k(rows):
    """rank-sum of silhouette and stability; ties to the smaller k."""
    sil_rank = {r["k"]: i for i, r in enumerate(sorted(rows, key=lambda r: -r["silhouette"]))}
    st_rank = {r["k"]: i for i, r in enumerate(sorted(rows, key=lambda r: -r["stability_ari"]))}
    return min(rows, key=lambda r: (sil_rank[r["k"]] + st_rank[r["k"]], r["k"]))["k"]


def hdbscan_fit(Z, min_cluster_size):
    from sklearn.cluster import HDBSCAN
    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5).fit_predict(Z)


def nmf_sweep(X, seed):
    from sklearn.decomposition import NMF
    from sklearn.metrics import adjusted_rand_score
    rows, labels, models = [], {}, {}
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    for k in NMF_KS:
        m = NMF(n_components=k, init="nndsvda", random_state=seed, max_iter=400)
        W = m.fit_transform(X); lab = W.argmax(1)
        aris = []
        for b in range(N_BOOT):
            idx = rng.choice(n, size=int(0.8 * n), replace=False)
            mb = NMF(n_components=k, init="nndsvda", random_state=seed + b + 1, max_iter=400)
            lb = mb.fit_transform(X[idx]).argmax(1)
            aris.append(adjusted_rand_score(lab[idx], lb))
        Wn = W / (W.sum(1, keepdims=True) + 1e-9)
        rows.append({"k": k, "silhouette": round(silhouette(Wn, lab), 4), "stability_ari": round(float(np.mean(aris)), 4),
                     "stability_sd": round(float(np.std(aris)), 4), "recon_err": round(float(m.reconstruction_err_), 4),
                     "sizes": "/".join(str(c) for _, c in sorted(Counter(lab).items()))})
        labels[k] = lab; models[k] = m
    return rows, labels, models


# ---------------------------------------------------------------- description -------
def keyness_terms(texts, lab, top=8):
    """weighted log-odds (informative Dirichlet prior) of cluster vs rest, top terms per cluster."""
    cnt = defaultdict(Counter)
    for text, c in zip(texts, lab):
        cnt[c].update(analyzer(text))
    allc = Counter()
    for c in cnt.values():
        allc.update(c)
    N = sum(allc.values()); a0 = 500.0
    out = {}
    for c, cc in cnt.items():
        n1 = sum(cc.values()); n2 = N - n1
        rows = []
        for w, tot in allc.items():
            if tot < 5:
                continue
            aw = a0 * tot / N; y1 = cc[w]; y2 = tot - y1
            d = math.log((y1 + aw) / (n1 + a0 - y1 - aw)) - math.log((y2 + aw) / (n2 + a0 - y2 - aw))
            z = d / math.sqrt(1 / (y1 + aw) + 1 / (y2 + aw))
            rows.append((z, w))
        rows.sort(reverse=True)
        out[int(c)] = [w for _, w in rows[:top]]
    return out


def purity(true, pred):
    ct = defaultdict(Counter)
    for t, p in zip(true, pred):
        ct[p][t] += 1
    return sum(c.most_common(1)[0][1] for c in ct.values()) / max(len(true), 1)


def external(lab, true):
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    mask = np.array([t is not None for t in true])
    if mask.sum() < 10 or len(set(lab[mask])) < 2:
        return {"n": int(mask.sum()), "ari": None, "nmi": None, "purity": None}
    tt = [true[i] for i in np.where(mask)[0]]; ll = lab[mask]
    return {"n": int(mask.sum()), "ari": round(float(adjusted_rand_score(tt, ll)), 3),
            "nmi": round(float(normalized_mutual_info_score(tt, ll)), 3), "purity": round(purity(tt, ll), 3)}


def describe(turns, lab, style, texts, S):
    out = []
    kt = keyness_terms(texts, lab)
    ci = STYLE_COLS.index
    for c in sorted(set(int(x) for x in lab)):
        idx = np.where(lab == c)[0]
        roles = Counter(turns[i]["role"] for i in idx); bodies = Counter(turns[i]["body"] for i in idx)
        out.append({
            "cluster": c, "n": len(idx), "share": round(len(idx) / len(lab), 3),
            "public_share": round(roles["public"] / len(idx), 3),
            "top_bodies": "; ".join(f"{b} {n}" for b, n in bodies.most_common(3)),
            "n_bodies": len(bodies),
            "mean_words": round(float(np.exp(style[idx, ci("log_words")]).mean()), 1),
            "mean_compound": round(float(style[idx, ci("vader_compound")].mean()), 3),
            "hostility_per100w": round(float(style[idx, ci("hostility_r")].mean()), 3),
            "we_minus_i": round(float(style[idx, ci("we_minus_i")].mean()), 2),
            "question_r": round(float(style[idx, ci("question_r")].mean()), 1),
            "top_terms": ", ".join(kt.get(c, [])),
            "examples": examples(turns, idx, S, lab, c),
        })
    return out


def examples(turns, idx, S, lab, c, n=3, cut=40):
    cen = S[idx].mean(0)
    d = np.linalg.norm(S[idx] - cen, axis=1)
    order = idx[np.argsort(d)]
    ex, seen_eps = [], set()
    for i in order:
        t = turns[i]
        if t["episode_id"] in seen_eps:
            continue
        seen_eps.add(t["episode_id"])
        w = clean_text(t["text"]).split()
        ex.append({"episode_id": t["episode_id"], "turn_idx": t["turn_idx"], "role": t["role"], "body": t["body"],
                   "text": " ".join(w[:cut]) + (" ..." if len(w) > cut else "")})
        if len(ex) == n:
            break
    return ex


# ---------------------------------------------------------------- figures -----------
def project2d(Z, seed):
    try:
        import umap
        return umap.UMAP(n_components=2, random_state=seed, n_neighbors=20, min_dist=0.1).fit_transform(Z), "UMAP"
    except Exception as e:  # noqa: BLE001
        from sklearn.decomposition import PCA
        print("umap unavailable:", e)
        return PCA(n_components=2, random_state=seed).fit_transform(Z), "PCA"


def cat_colors(values, palette):
    cats = sorted(set(v for v in values if v is not None), key=str)
    cmap = {c: palette[i % len(palette)] for i, c in enumerate(cats)}
    return [cmap.get(v, "#c8c8c8") for v in values], cmap


def fig_projection(P2, proj_name, lab, roles, llm3, facet, view, out):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    pal = ["#1b6ca8", "#e07b39", "#3a9d5d", "#b23a48", "#7b5ea7", "#c9a227", "#4aa3a2", "#8c6d4f"]
    fig, axes = ap.figure(1, 3, f"Unsupervised clusters of speaker turns -- {view} view, locate route = {facet}",
                          f"{proj_name} of the {view} feature space; same points coloured three ways. EXPLORATORY.", (16, 6))
    panels = [("k-means cluster", lab), ("role (dais / public)", roles), ("Sonnet NIMBY/YIMBY label (300-turn sample; grey = unlabelled)", llm3)]
    for ax, (title, vals) in zip(axes[0], panels):
        cols, cmap = cat_colors(list(vals), pal)
        grey = np.array([v is None for v in vals])
        if grey.any():
            ax.scatter(P2[grey, 0], P2[grey, 1], s=6, c="#d9d9d9", alpha=0.5, linewidths=0)
        keep = ~grey
        ax.scatter(P2[keep, 0], P2[keep, 1], s=9 if title.startswith("Sonnet") else 7, c=[c for c, g in zip(cols, grey) if not g], alpha=0.75, linewidths=0)
        ap.style(ax, ygrid=False); ap.panel_title(ax, title); ax.set_xticks([]); ax.set_yticks([])
        ap.legend(ax, [Line2D([], [], marker="o", ls="", color=c) for c in cmap.values()], [str(k) for k in cmap], loc="lower left")
    ap.footnote(fig, "EXPLORATORY. Turns >= 25 words, density/both episodes, roles_source = db, anonymous per-episode speaker clusters "
                     "(DECISIONS #15). Clusters are unsupervised; labels are used only to colour, never to fit.")
    ap.save(fig, out, rect=(0, 0.06, 1, 0.9))


def fig_heatmap(turns, lab, facet, view, out):
    import matplotlib.pyplot as plt
    bodies = [b for b, _ in Counter(t["body"] for t in turns).most_common()]
    ks = sorted(set(int(x) for x in lab))
    M = np.zeros((len(bodies), len(ks)))
    for t, c in zip(turns, lab):
        M[bodies.index(t["body"]), ks.index(int(c))] += 1
    R = M / np.maximum(M.sum(1, keepdims=True), 1)
    fig, axes = ap.figure(1, 1, f"Cluster by body -- {view} view, locate route = {facet}",
                          "Row-normalised share of each body's turns in each cluster (n per row at right). EXPLORATORY.", (10, 0.45 * len(bodies) + 3))
    ax = axes[0][0]
    im = ax.imshow(R, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_yticks(range(len(bodies))); ax.set_yticklabels([f"{b}  (n={int(M[i].sum())})" for i, b in enumerate(bodies)])
    ax.set_xticks(range(len(ks))); ax.set_xticklabels([f"c{k}" for k in ks])
    for i in range(len(bodies)):
        for j in range(len(ks)):
            if R[i, j] >= 0.08:
                ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if R[i, j] > 0.55 else "#222")
    ap.style(ax, ygrid=False)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ap.footnote(fig, "EXPLORATORY. If clusters tracked stance, rows would be mixed; blocks along a body mean the clusters track the body (its agenda, its names, its register).")
    ap.save(fig, out, rect=(0, 0.06, 1, 0.9))


# ---------------------------------------------------------------- main --------------
def run_facet(facet, turns, llm, va, hpats, fpats, fh, outdir):
    seed = SEED
    texts = [t["text"] for t in turns]
    roles = np.array([t["role"] for t in turns]); bodies = np.array([t["body"] for t in turns])
    llm3 = [llm.get(t["turn_id"], {}).get("label3") for t in turns]
    n_llm = sum(1 for x in llm3 if x)
    log(f"[{facet}] {len(turns)} turns, {len(set(t['episode_id'] for t in turns))} episodes, {len(set(bodies))} bodies, {n_llm} LLM-labelled", fh)

    X, Zlex, vec = tfidf_view(texts, seed)
    log(f"[{facet}] tfidf vocab {X.shape[1]} terms, LSA {Zlex.shape[1]} d", fh)
    from sklearn.preprocessing import StandardScaler
    style = style_view(turns, va, hpats, fpats)
    Zsty = StandardScaler().fit_transform(style)

    res = {"facet": facet, "n_turns": len(turns), "n_episodes": len(set(t["episode_id"] for t in turns)), "n_bodies": int(len(set(bodies))),
           "n_llm_labelled": n_llm, "role_counts": dict(Counter(roles.tolist())), "views": {}}
    projections = {}
    for view, Z in (("lex", Zlex), ("style", Zsty)):
        rows, labels = sweep_kmeans(Z, seed)
        k = pick_k(rows); lab = labels[k]
        log(f"[{facet}] {view}: " + "; ".join(f"k{r['k']} sil {r['silhouette']:.3f} stab {r['stability_ari']:.2f}" for r in rows) + f" -> k={k}", fh)
        hd = hdbscan_fit(Z, max(15, len(turns) // 60))
        hd_n = len(set(hd) - {-1}); noise = float((hd == -1).mean())
        hd_ext = {}
        if hd_n >= 2:
            m = hd != -1
            hd_ext = {"n_clusters": hd_n, "noise_share": round(noise, 3), "silhouette_non_noise": round(silhouette(Z[m], hd[m]), 4),
                      "vs_role": external(hd[m], roles[m].tolist()), "vs_body": external(hd[m], bodies[m].tolist()),
                      "vs_llm3": external(hd[m], [llm3[i] for i in np.where(m)[0]])}
        else:
            hd_ext = {"n_clusters": hd_n, "noise_share": round(noise, 3)}
        log(f"[{facet}] {view}: HDBSCAN {hd_n} clusters, noise {noise:.2f}", fh)
        res["views"][view] = {
            "sweep": rows, "best_k": int(k),
            "validation": {"vs_llm3": external(lab, llm3), "vs_role": external(lab, roles.tolist()), "vs_body": external(lab, bodies.tolist())},
            "validation_all_k": {int(kk): {"vs_llm3": external(labels[kk], llm3), "vs_role": external(labels[kk], roles.tolist()),
                                           "vs_body": external(labels[kk], bodies.tolist())} for kk in KS},
            "hdbscan": hd_ext,
            "clusters": describe(turns, lab, style, texts, Z),
        }
        P2, pname = project2d(Z, seed)
        projections[view] = (P2, pname)
        fig_projection(P2, pname, [f"c{x}" for x in lab], roles.tolist(), llm3, facet, view, outdir / f"proj_{view}_{facet}.png")
        fig_heatmap(turns, lab, facet, view, outdir / f"heat_{view}_{facet}.png")
        with open(outdir / f"assign_{view}_{facet}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["turn_id", "episode_id", "turn_idx", "role", "body", "cluster", "hdbscan", "x", "y", "llm3"])
            for t, c, h, p, l in zip(turns, lab, hd, P2, llm3):
                w.writerow([t["turn_id"], t["episode_id"], t["turn_idx"], t["role"], t["body"], int(c), int(h), round(float(p[0]), 3), round(float(p[1]), 3), l or ""])

    # NMF topics as a fourth view
    rows, labels, models = nmf_sweep(X, seed)
    k = pick_k(rows); lab = labels[k]
    log(f"[{facet}] nmf: " + "; ".join(f"k{r['k']} sil {r['silhouette']:.3f} stab {r['stability_ari']:.2f}" for r in rows) + f" -> k={k}", fh)
    terms = np.array(vec.get_feature_names_out())
    topics = {int(i): ", ".join(terms[np.argsort(-comp)[:8]]) for i, comp in enumerate(models[k].components_)}
    desc = describe(turns, lab, style, texts, Zlex)
    for d in desc:
        d["nmf_top_terms"] = topics[d["cluster"]]
    res["views"]["nmf"] = {
        "sweep": rows, "best_k": int(k),
        "validation": {"vs_llm3": external(lab, llm3), "vs_role": external(lab, roles.tolist()), "vs_body": external(lab, bodies.tolist())},
        "clusters": desc,
    }
    P2, pname = projections["lex"]
    fig_projection(P2, pname, [f"t{x}" for x in lab], roles.tolist(), llm3, facet, "nmf", outdir / f"proj_nmf_{facet}.png")
    fig_heatmap(turns, lab, facet, "nmf", outdir / f"heat_nmf_{facet}.png")

    # how well does role/body alone predict a cluster? baseline: labels = role
    # reference points: how much the external labels themselves overlap (LLM stance vs role / body)
    res["baseline"] = {"llm3_vs_role": external(roles, llm3), "llm3_vs_body": external(bodies, llm3)}
    return res


def main(argv=None):
    pa = argparse.ArgumentParser()
    pa.add_argument("cmd", choices=["run"])
    pa.add_argument("--facets", default="classifier,agenda_index,local_asr,swagit_transcript")
    pa.add_argument("--out", default=str(OUT))
    a = pa.parse_args(argv)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    fh = open(outdir / "pilot.log", "a")
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    va = SentimentIntensityAnalyzer()
    hpats = hostility_pats()
    fpats = {k: [at.term_pattern(t) for t in ts] for k, ts in at.load_frames(ROOT / "plan" / "frames.yaml").items()}
    turns, missing, n_eps = load_turns()
    llm = load_llm_labels()
    log(f"[load] {len(turns)} turns >= {MIN_WORDS} words, density/both, dais/public, from {n_eps} episodes; {len(missing)} redacted missing; "
        f"{len(llm)} LLM labels available ({'file present' if llm else 'llm_labels.jsonl absent'})", fh)
    by_facet = defaultdict(list)
    dropped = Counter()
    for t in turns:
        if t["roles_source"] != "db":
            dropped[t["roles_source"]] += 1; continue
        by_facet[t["method"]].append(t)
    log(f"[load] roles_source != db dropped (not pooled): {dict(dropped)}; facets: {{{', '.join(f'{k}: {len(v)}' for k, v in by_facet.items())}}}", fh)
    ap.SCOPE = ""  # we stamp the facet in every title/footnote ourselves
    results = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_turns_total": len(turns), "llm_labels": len(llm),
               "views": {"lex": "TF-IDF uni+bigrams -> LSA 100-d", "style": STYLE_COLS, "embed": "skipped: no local sentence-embedding model installed",
                         "nmf": "NMF topics on TF-IDF, argmax"},
               "facets": {}}
    for facet in a.facets.split(","):
        ts = by_facet.get(facet, [])
        if len(ts) < MIN_FACET_TURNS:
            log(f"[{facet}] skipped: {len(ts)} turns < {MIN_FACET_TURNS}", fh); continue
        results["facets"][facet] = run_facet(facet, ts, llm, va, hpats, fpats, fh, outdir)
        (outdir / "results.json").write_text(json.dumps(results, indent=1, default=str))
    log("[done]", fh)


if __name__ == "__main__":
    main()
