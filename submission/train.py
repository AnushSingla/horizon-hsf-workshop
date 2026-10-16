#!/usr/bin/env python3
"""Train a signal/background classifier on the pool and export it as model.json (text) for fit.py.

    python3 train.py [pool.npz] [--out model.json] [--epochs 80] [--naug 3] [--no-aug]

A small multilayer perceptron trained with weighted binary cross-entropy by AdamW (cosine LR
with warmup) on the EVEN-indexed pool events only.  Signal is replaced by several mass-morphed
copies so the score is trained across the mass hypotheses the fit scans.  The train/validation
split is made on the original events BEFORE augmentation.  Numpy only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H  # noqa: E402
from features import FEATURES, features  # noqa: E402

TRAIN_MASS_GRID = np.arange(120.0, 130.01, 1.0)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def _forward(Ws, bs, X):
    a = X
    for i, (W, b) in enumerate(zip(Ws, bs)):
        a = a @ W + b
        if i < len(Ws) - 1:
            a = np.maximum(a, 0.0)
    return a


def morph_signal(sig, rng, grid):
    """Morph each signal event to a random grid mass."""
    ms = rng.choice(grid, size=len(sig["weight"]))
    out = {k: v.copy() for k, v in sig.items()}
    for m in np.unique(ms):
        sel = ms == m
        mm = H.morph(H.select(sig, sel), m)
        for c in H.SCALED:
            out[c][sel] = mm[c]
    return out


def build_set(events, rng, aug, naug):
    """Signal is replaced by `naug` morphed copies (weights / naug); background unchanged."""
    if not aug:
        return events
    is_sig = events["label"] == 1
    bkg = H.select(events, np.where(~is_sig)[0])
    sig = H.select(events, np.where(is_sig)[0])
    parts = [bkg]
    for _ in range(naug):
        c = morph_signal(sig, rng, TRAIN_MASS_GRID)
        c["weight"] = c["weight"] / naug
        parts.append(c)
    return {k: np.concatenate([p[k] for p in parts]) for k in events}


def train_mlp(X, y, w, Xv, yv, wv, hidden=(64, 64), epochs=80, batch=2048, lr=2e-3, wd=1e-4,
              patience=10, warmup=2, seed=1):
    """Weighted BCE, AdamW, cosine LR, early stopping on weighted validation loss."""
    rng = np.random.default_rng(seed)
    sizes = [X.shape[1], *hidden, 1]
    Ws = [rng.normal(0.0, np.sqrt(2.0 / a), (a, b)) for a, b in zip(sizes[:-1], sizes[1:])]
    bs = [np.zeros(b) for b in sizes[1:]]
    m = [np.zeros_like(p) for p in Ws + bs]
    v = [np.zeros_like(p) for p in Ws + bs]
    t, best = 0, (np.inf, None, 0)
    steps_per_ep = int(np.ceil(len(y) / batch))
    total = epochs * steps_per_ep

    def val_loss():
        p = np.clip(_sigmoid(_forward(Ws, bs, Xv))[:, 0], 1e-7, 1 - 1e-7)
        return float(-np.sum(wv * (yv * np.log(p) + (1.0 - yv) * np.log(1.0 - p))) / wv.sum())

    ep = 0
    for ep in range(epochs):
        order = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            j = order[s:s + batch]
            xb, yb, wb = X[j], y[j], w[j]
            acts = [xb]
            for i, (W, b) in enumerate(zip(Ws, bs)):
                z = acts[-1] @ W + b
                acts.append(np.maximum(z, 0.0) if i < len(Ws) - 1 else z)
            d = ((_sigmoid(acts[-1][:, 0]) - yb) * (wb / wb.sum()))[:, None]
            gW, gb = [None] * len(Ws), [None] * len(Ws)
            for i in range(len(Ws) - 1, -1, -1):
                gW[i] = acts[i].T @ d
                gb[i] = d.sum(0)
                if i > 0:
                    d = (d @ Ws[i].T) * (acts[i] > 0)
            t += 1
            if t <= warmup * steps_per_ep:
                lr_t = lr * t / (warmup * steps_per_ep)
            else:
                lr_t = 0.5 * lr * (1 + np.cos(np.pi * (t - warmup * steps_per_ep)
                                              / max(1, total - warmup * steps_per_ep)))
            params, grads = Ws + bs, gW + gb
            for k in range(len(params)):
                m[k] = 0.9 * m[k] + 0.1 * grads[k]
                v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                step = (m[k] / (1.0 - 0.9 ** t)) / (np.sqrt(v[k] / (1.0 - 0.999 ** t)) + 1e-8)
                if k < len(Ws):                       # decoupled decay on weights only
                    params[k] -= lr_t * wd * params[k]
                params[k] -= lr_t * step
        loss = val_loss()
        if loss < best[0] - 1e-5:
            best = (loss, ([W.copy() for W in Ws], [b.copy() for b in bs]), ep)
        elif ep - best[2] >= patience:
            break
    return best[1][0], best[1][1], ep + 1, best[0]


def weighted_auc(y, s, w):
    o = np.argsort(s, kind="mergesort")
    y, w = y[o], w[o]
    neg = np.cumsum(w * (1.0 - y))
    return float(np.sum(w * y * neg) / ((w * y).sum() * (w * (1.0 - y)).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pool", nargs="?", default=str(HERE.parent.parent / "datasets" / "public" / "seed_000" / "mc" / "pool.npz"))
    ap.add_argument("--out", default=str(HERE / "model.json"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--no-aug", action="store_true", help="disable mass-morphed signal")
    ap.add_argument("--naug", type=int, default=3, help="morphed copies per signal event")
    ap.add_argument("--hidden", default="64,64")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    aug = not a.no_aug
    hidden = tuple(int(x) for x in a.hidden.split(","))

    pool = H.load(a.pool)
    N = len(pool["label"])
    parity = np.arange(N) % 2
    even = H.select(pool, parity == 0)
    held = H.select(pool, parity == 1)
    rng = np.random.default_rng(a.seed)

    # split on ORIGINAL events first, so morphed copies never straddle train/val
    n0 = len(even["label"])
    perm = rng.permutation(n0)
    va_idx, tr_idx = np.sort(perm[:int(0.15 * n0)]), np.sort(perm[int(0.15 * n0):])
    tr = build_set(H.select(even, tr_idx), rng, aug, a.naug)
    va = build_set(H.select(even, va_idx), rng, aug, 1)

    ytr, wtr = tr["label"].astype(float), tr["weight"].astype(float)
    yva, wva = va["label"].astype(float), va["weight"].astype(float)

    # clip at the 99th percentile (train statistics) and balance classes
    wtr_c, wva_c = wtr.copy(), wva.copy()
    for c in (0, 1):
        cap = np.quantile(wtr[ytr == c], 0.99)
        wtr_c[ytr == c] = np.minimum(wtr_c[ytr == c], cap)
        wva_c[yva == c] = np.minimum(wva_c[yva == c], cap)
    for c in (0, 1):
        norm = 0.5 / wtr_c[ytr == c].sum()
        wtr_c[ytr == c] *= norm
        wva_c[yva == c] *= norm
    wtr_c *= len(ytr)
    wva_c *= len(ytr)

    Xtr, Xva = features(tr), features(va)
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-9
    Xtr, Xva = (Xtr - mean) / std, (Xva - mean) / std
    Ws, bs, ne, loss = train_mlp(Xtr, ytr, wtr_c, Xva, yva, wva_c, hidden=hidden,
                                 epochs=a.epochs, seed=a.seed)

    # held-out diagnostics on odd events (diagnostic only; never tune on them)
    p = _sigmoid(_forward(Ws, bs, (features(held) - mean) / std)[:, 0])
    p_tr = _sigmoid(_forward(Ws, bs, Xtr)[:, 0])
    yl, wl = held["label"].astype(int), held["weight"].astype(float) * 2.0
    print(f"trained {ne} epochs (aug={aug}x{a.naug}, val loss {loss:.5f}); weighted AUC: "
          f"train {weighted_auc(ytr, p_tr, wtr_c):.4f}, held-out {weighted_auc(yl.astype(float), p, wl):.4f}")
    edges = [np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]), H.MT_EDGES]
    mt = H.kinematics(held)["mt"]
    hs, _, _ = np.histogram2d(p[yl == 1], mt[yl == 1], edges, weights=wl[yl == 1])
    hb, _, _ = np.histogram2d(p[yl == 0], mt[yl == 0], edges, weights=wl[yl == 0])
    print(f"expected Asimov Z at 125 GeV, (score, m_T) bins: {H.asimov_significance(hs, hb):.2f}")

    model = {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(), "activation": "relu",
             "augmented": bool(aug),
             "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in zip(Ws, bs)]}
    Path(a.out).write_text(json.dumps(model))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()