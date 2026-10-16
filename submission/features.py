"""The event features the classifier sees: FEATURES (names, in order) and features(ev) (the matrix).

train.py trains on them, fit.py evaluates the saved network on them, so the two must use this one file.
Imports only numpy and hww (no scikit-learn), as the contract requires.
"""
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
    """Absolute azimuthal angle between a and b, wrapped to [0, pi]."""
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