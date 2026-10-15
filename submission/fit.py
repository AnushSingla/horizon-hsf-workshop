#!/usr/bin/env python3
"""The classifier plugged into the measurement: a template fit in bins of network score and transverse mass.

    python3 fit.py <dataset_dir> <output_dir>      (run.sh calls this; model.json from train.py sits beside it)

The network (model.json) is evaluated with numpy on the data and on the pool. Templates live on the
(score, m_T) plane: the score axis buys purity, m_T carries the mass. The signal template at mass m is the
morphed 125 GeV sample scored by the same network, so the classifier's response to a heavier or lighter Higgs
is part of the model. Templates use only the odd-indexed pool events (train.py trains on the even ones).
Fit, errors and significance as in examples/template_fit.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hww as H  # noqa: E402
from train import features  # noqa: E402

SCORE_EDGES = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
M_LO, M_HI = float(H.MASS_GRID[0]), float(H.MASS_GRID[-1])


def load_model(path):
    m = json.loads(Path(path).read_text())
    return {"mean": np.array(m["mean"]), "std": np.array(m["std"]), "layers": [(np.array(l["W"]), np.array(l["b"])) for l in m["layers"]]}


def score(model, ev):
    x = (features(ev) - model["mean"]) / model["std"]
    for i, (W, b) in enumerate(model["layers"]):
        x = x @ W + b
        if i < len(model["layers"]) - 1:
            x = np.maximum(x, 0.0)
    return 1.0 / (1.0 + np.exp(-x[:, 0]))


def template(model, ev, weights=None):
    h, _, _ = np.histogram2d(score(model, ev), H.kinematics(ev)["mt"], [SCORE_EDGES, H.MT_EDGES], weights=weights)
    return h


def main(ds: Path, out: Path):
    model = load_model(HERE / "model.json")
    data = H.load(ds / "data" / "events.npz"); pool = H.load(ds / "mc" / "pool.npz")
    odd = np.arange(len(pool["label"])) % 2 == 1      # train.py trains on the even events only
    pool = H.select(pool, odd); pool["weight"] = pool["weight"] * 2.0
    sig, bkg = H.select(pool, pool["label"] == 1), H.select(pool, pool["label"] == 0)
    n = template(model, data)
    S = {float(m): template(model, H.morph(sig, m), sig["weight"]) for m in H.MASS_GRID}
    B = template(model, bkg, bkg["weight"])

    def nll(m, mu, b):
        lam = np.clip(mu * H.interpolate(S, m) + b * B, 1e-9, None)
        return float(np.sum(lam - n * np.log(lam)))

    def profile(m):
        r = minimize(lambda p: nll(m, p[0], p[1]), [1.0, 1.0], method="L-BFGS-B", bounds=[(-2.0, 20.0), (0.2, 5.0)])
        return r.fun, r.x

    scan = [(profile(m), m) for m in sorted(S)]
    (f0, (mu0, b0)), m0 = min(scan, key=lambda t: t[0][0])
    r = minimize(lambda p: nll(np.clip(p[0], M_LO, M_HI), p[1], p[2]), [m0, mu0, b0], method="Nelder-Mead",
                 options={"xatol": 1e-3, "fatol": 1e-6, "maxiter": 4000})
    m_hat, mu_hat, b_hat = float(np.clip(r.x[0], M_LO, M_HI)), float(r.x[1]), float(r.x[2])
    f_min = nll(m_hat, mu_hat, b_hat)
    x = np.array([m_hat, mu_hat, b_hat]); h = np.array([0.5, 0.02, 0.01])
    hess = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            ei, ej = np.eye(3)[i] * h[i], np.eye(3)[j] * h[j]
            hess[i, j] = (nll(*(x + ei + ej)) - nll(*(x + ei - ej)) - nll(*(x - ei + ej)) + nll(*(x - ei - ej))) / (4 * h[i] * h[j])
    try:
        err = np.sqrt(np.diag(np.linalg.inv(hess)))
        if not np.all(np.isfinite(err)):
            raise np.linalg.LinAlgError
    except np.linalg.LinAlgError:
        err = np.array([0.5 * (M_HI - M_LO), 1.0, 0.1])
    mu_err = float(max(err[1], 1e-3))
    ms = np.arange(M_LO, M_HI + 0.01, 1.0)
    prof = np.array([profile(m)[0] for m in ms]); prof = prof - min(prof.min(), f_min)
    center = ms[int(np.argmin(np.abs(ms - m_hat)))]
    lo, hi = M_LO, M_HI
    left = [(m, v) for m, v in zip(ms, prof) if m < center]
    for (m1, v1), (m2, v2) in zip(left[:-1], left[1:]):
        if v1 >= 0.5 > v2:
            lo = m1 + (v1 - 0.5) / (v1 - v2) * (m2 - m1)
    right = [(m, v) for m, v in zip(ms, prof) if m > center]
    for (m1, v1), (m2, v2) in zip(right[:-1], right[1:]):
        if v1 < 0.5 <= v2:
            hi = m1 + (0.5 - v1) / (v2 - v1) * (m2 - m1)
            break
    m_err = float(np.clip(0.5 * (hi - lo), 0.5, 0.5 * (M_HI - M_LO)))

    r0 = minimize(lambda p: nll(m_hat, 0.0, p[0]), [b_hat], method="L-BFGS-B", bounds=[(0.2, 5.0)])
    z = float(np.sqrt(max(0.0, 2.0 * (r0.fun - f_min)))) if mu_hat > 0 else 0.0
    result = {"pois": [{"name": "mH", "value": m_hat, "stat": m_err, "syst": 0.0, "total": m_err},
                       {"name": "mu", "value": mu_hat, "stat": mu_err, "syst": 0.0, "total": mu_err}],
              "significance_obs": z, "discovery": bool(z >= 5.0),
              "note": f"template fit in (network score, m_T); mass interval [{lo:.1f}, {hi:.1f}]; background normalisation {b_hat:.3f}"}
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULT.json").write_text(json.dumps(result, indent=1))
    print(f"mH = {m_hat:.2f} ± {m_err:.2f}  mu = {mu_hat:.3f} ± {mu_err:.3f}  b = {b_hat:.3f}  Z = {z:.2f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))