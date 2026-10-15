#!/usr/bin/env python3
"""Train a signal-vs-background network on the pool and export it as text for the pipeline.

    python3 train.py [datasets/public/seed_000/mc/pool.npz] [--out model.json] [--aug]

A multilayer perceptron (two hidden layers of 64), trained with numpy alone on the EVEN-indexed simulated
events with their labels, written to model.json: the feature list, the standardisation, the weight matrices.
fit.py builds its templates from the ODD-indexed events only, so the templates are never made from events
the network was trained on. fit.py evaluates the network with numpy alone, as the contract requires.

What this version adds to the scikit-learn one:
  * five more features (the transverse mass of each lepton with the missing ET, the angle between the
    dilepton system and the missing ET, both lepton |eta|), 21 in all;
  * training with the event weights: each class clipped at its 99th weight percentile so a few huge
    weights do not dominate, then the two classes balanced, so the score is tuned to the mixture that is
    measured, on all the even events instead of a subsample;
  * early stopping on the weighted validation loss, and no scikit-learn needed to train.

--aug morphs each training signal event to a random grid mass, so the score is trained across the
mass hypotheses the fit scans.

fit.py imports `features` from this file at run time: keep that function, and keep the imports at the top light.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H  # noqa: E402

FEATURES = ["lep_pt_0", "lep_pt_1", "met_et", "dphi", "mll", "ptll", "mt", "jet_n", "deta",
            "same_flavour", "opp_sign", "dphi_met_ll", "pt_sum", "jet_pt_0", "mll_over_mt", "ptll_over_mt",
            "mt_l0", "mt_l1", "dphi_ll_met", "abs_eta_0", "abs_eta_1"]


def _dphi(a, b):
    return np.abs(np.mod(a - b + 3 * np.pi, 2 * np.pi) - np.pi)


def features(ev):
    """The feature matrix of a set of events, in the order of FEATURES."""
    k = H.kinematics(ev)
    f = lambda c: np.asarray(ev[c], float)
    pt0, pt1, phi0, phi1 = f("lep_pt_0"), f("lep_pt_1"), f("lep_phi_0"), f("lep_phi_1")
    met, mphi = f("met_et"), f("met_phi")
    d0, d1 = _dphi(mphi, phi0), _dphi(mphi, phi1)
    px, py = pt0 * np.cos(phi0) + pt1 * np.cos(phi1), pt0 * np.sin(phi0) + pt1 * np.sin(phi1)
    cols = {"lep_pt_0": pt0, "lep_pt_1": pt1, "met_et": met,
            "dphi": k["dphi"], "mll": k["mll"], "ptll": k["ptll"], "mt": k["mt"], "jet_n": f("jet_n"),
            "deta": np.abs(f("lep_eta_0") - f("lep_eta_1")),
            "same_flavour": (ev["lep_type_0"] == ev["lep_type_1"]).astype(float),
            "opp_sign": (f("lep_charge_0") * f("lep_charge_1") < 0).astype(float),
            "dphi_met_ll": np.minimum(d0, d1),
            "pt_sum": pt0 + pt1,
            "jet_pt_0": f("jet_pt_0"),
            "mll_over_mt": k["mll"] / np.maximum(k["mt"], 1.0),
            "ptll_over_mt": k["ptll"] / np.maximum(k["mt"], 1.0),
            "mt_l0": np.sqrt(np.clip(2 * pt0 * met * (1 - np.cos(d0)), 0, None)),
            "mt_l1": np.sqrt(np.clip(2 * pt1 * met * (1 - np.cos(d1)), 0, None)),
            "dphi_ll_met": _dphi(np.arctan2(py, px), mphi),
            "abs_eta_0": np.abs(f("lep_eta_0")), "abs_eta_1": np.abs(f("lep_eta_1"))}
    return np.column_stack([cols[n] for n in FEATURES])


def augment(sig, rng):
    """Morph each signal event to a random grid mass, so the score is trained across the mass hypotheses."""
    ms = rng.choice(H.MASS_GRID, size=len(sig["weight"]))
    out = {k: v.copy() for k, v in sig.items()}
    for m in np.unique(ms):
        sel = ms == m
        mm = H.morph(H.select(sig, sel), m)
        for c in H.SCALED:
            out[c][sel] = mm[c]
    return out


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _forward(Ws, bs, X):
    acts = [X]
    for i, (W, b) in enumerate(zip(Ws, bs)):
        a = acts[-1] @ W + b
        acts.append(np.maximum(a, 0.0) if i < len(Ws) - 1 else a)
    return acts


def fit_mlp(X, y, w, Xv, yv, wv, hidden=(64, 64), epochs=40, batch=2048, lr=2e-3, l2=1e-5, patience=5, seed=1):
    """Weighted binary cross-entropy, Adam, early stopping on the weighted validation loss."""
    rng = np.random.default_rng(seed)
    sizes = [X.shape[1], *hidden, 1]
    Ws = [rng.normal(0, np.sqrt(2 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
    bs = [np.zeros(b) for b in sizes[1:]]
    m = [np.zeros_like(p) for p in Ws + bs]
    v = [np.zeros_like(p) for p in Ws + bs]
    t, best = 0, (np.inf, None, 0)

    def val_loss():
        p = np.clip(_sigmoid(_forward(Ws, bs, Xv)[-1][:, 0]), 1e-7, 1 - 1e-7)
        return -np.sum(wv * (yv * np.log(p) + (1 - yv) * np.log(1 - p))) / wv.sum()

    for ep in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            j = order[s:s + batch]
            acts = _forward(Ws, bs, X[j])
            d = ((_sigmoid(acts[-1][:, 0]) - y[j]) * w[j] / w[j].sum())[:, None]
            gW, gb = [None] * len(Ws), [None] * len(Ws)
            for i in range(len(Ws) - 1, -1, -1):
                gW[i], gb[i] = acts[i].T @ d + l2 * Ws[i], d.sum(0)
                if i > 0:
                    d = (d @ Ws[i].T) * (acts[i] > 0)
            t += 1
            params, grads = Ws + bs, gW + gb
            for k in range(len(params)):
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                params[k] -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
        loss = val_loss()
        if loss < best[0] - 1e-5:
            best = (loss, ([W.copy() for W in Ws], [b.copy() for b in bs]), ep)
        elif ep - best[2] >= patience:
            break
    return best[1][0], best[1][1], ep + 1


def weighted_auc(y, s, w):
    o = np.argsort(s, kind="mergesort")
    y, w = y[o], w[o]
    neg = np.cumsum(w * (1 - y))
    return float(np.sum(w * y * neg) / ((w * y).sum() * (w * (1 - y)).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", nargs="?", default=str(HERE.parent.parent / "datasets" / "public" / "seed_000" / "mc" / "pool.npz"))
    ap.add_argument("--out", default=str(HERE / "model.json"))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--aug", action="store_true", help="train on mass-morphed signal")
    a = ap.parse_args()

    pool = H.load(a.pool)
    N = len(pool["label"])
    parity = np.arange(N) % 2
    rng = np.random.default_rng(1)
    train = H.select(pool, parity == 0)       # fit.py builds its templates from parity == 1
    held = H.select(pool, parity == 1)
    if a.aug:
        sig_i = np.where(train["label"] == 1)[0]
        aug = augment(H.select(train, sig_i), rng)
        for k in train:
            train[k] = train[k].copy()
            train[k][sig_i] = aug[k]

    y = train["label"].astype(float)
    w = train["weight"].astype(float)
    n = len(y)
    perm = rng.permutation(n)
    va, tr = perm[:int(0.15 * n)], perm[int(0.15 * n):]
    wt = w.copy()
    for c in (0, 1):                          # clip each class's weights, then balance the classes
        sel = y == c
        wt[sel] = np.minimum(wt[sel], np.quantile(w[sel], 0.99))
    is_tr = np.zeros(n, bool)
    is_tr[tr] = True
    for c in (0, 1):
        wt[y == c] *= 0.5 / wt[(y == c) & is_tr].sum()
    wt *= n

    X = features(train)
    mean, std = X[tr].mean(axis=0), X[tr].std(axis=0) + 1e-9
    Xs = (X - mean) / std
    Ws, bs, ne = fit_mlp(Xs[tr], y[tr], wt[tr], Xs[va], y[va], wt[va], epochs=a.epochs)

    # held-out check on the odd events: AUC and the Asimov Z of the (score, m_T) template at 125 GeV
    p = _sigmoid(_forward(Ws, bs, (features(held) - mean) / std)[-1][:, 0])
    yl, wl = held["label"].astype(int), held["weight"].astype(float) * 2.0
    p_tr = _sigmoid(_forward(Ws, bs, Xs[tr])[-1][:, 0])
    print(f"trained {ne} epochs on {len(tr)} events (aug={a.aug}); weighted AUC: train "
          f"{weighted_auc(y[tr], p_tr, w[tr]):.3f}, held-out {weighted_auc(yl.astype(float), p, wl):.3f} "
          f"(unweighted {weighted_auc(yl.astype(float), p, np.ones(len(yl))):.3f})")
    edges = [np.array([0, .2, .4, .6, .8, 1.0]), H.MT_EDGES]
    mt = H.kinematics(held)["mt"]
    hs, _, _ = np.histogram2d(p[yl == 1], mt[yl == 1], edges, weights=wl[yl == 1])
    hb, _, _ = np.histogram2d(p[yl == 0], mt[yl == 0], edges, weights=wl[yl == 0])
    print(f"expected Asimov Z at 125 GeV, (score, m_T) bins: {H.asimov_significance(hs, hb):.2f}")

    model = {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "activation": "relu",
             "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in zip(Ws, bs)]}
    Path(a.out).write_text(json.dumps(model))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()