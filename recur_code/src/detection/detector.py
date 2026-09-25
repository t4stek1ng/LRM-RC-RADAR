"""Deployment-side detector — the single implementation shared by the offline
replay (`replay_checkpoints.py`) and the online runtime (`detection_runtime.py`).

It owns exactly three things, all of which must be identical on both paths or
the online numbers are not comparable to the offline ones:

  1. **Feature construction at a checkpoint** (design §2.2 step 2). At total
     sequence length ``L`` the probe index into the per-gen-token arrays is
     ``i = L - prompt_len - 1``; the features are the *running* statistics over
     the gen segment ``[0, i]`` — ``cons_ind_mean`` (mean of
     ``attn_consistency_independent``) and ``pmf_slope`` (linear fit of
     ``prompt_mean_fraction``) — plus ``log L`` for the method-2 classifier.
     ``_running_mean`` / ``_running_slope`` are defined HERE and imported by
     the training driver, so training and deployment share one definition.

  2. **The τ_prob gate** (design §2.2 step 3). ``argmax_c p_c < τ`` ⇒ no call at
     this checkpoint, keep generating.

  3. **The trajectory-level trigger policy** (design §2.2 step 4 / §5). The
     deployed unit is "first confident attack call wins" — an online detector
     cannot majority-vote over checkpoints it has not reached yet. Benign calls
     never stop generation, so the trajectory verdict is the last decided class
     if no attack ever fired.

**Two ways to carry the classifier.** The trained artefact is the joblib bundle
written by ``run_model_detection_pipeline.py --stage train`` (StandardScaler + multinomial
LogisticRegression over ``[cons_ind_mean, pmf_slope, log_L]``), which needs
sklearn. The online runtime lives in the ``Recur`` env, which has vLLM/torch but
deliberately no sklearn — so this module can also load a **JSON export** of the
same model (scaler mean/scale + coefficients + intercepts) and run the softmax
in numpy. The export is bit-for-bit the same linear model; ``--export`` below
writes it and verifies max |Δp| against sklearn on random inputs.

Export (base env, has sklearn)::

    /root/miniconda3/bin/python recur_code/src/detection/detector.py \
        --bundle exp/detection_dataset/models/loop_detector_llama8b_4class_method2.joblib \
        --export exp/detection_dataset/models/loop_detector_llama8b_4class_method2.json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --- primitives shared with the training code --------------------------------
# These live here (not in the training driver) so the online runtime can import
# them in the `Recur` env, which has no sklearn. The training driver imports them
# back, so training and deployment keep exactly one definition.
CHECKPOINTS = [256, 512, 1024, 2048, 4096]
MIN_GEN_FOR_SLOPE = 3  # natural floor: np.polyfit(deg=1) + slope SE need >=3 pts.


def _running_mean(y: np.ndarray, up_to_idx: int) -> float:
    seg = y[: up_to_idx + 1]
    m = ~np.isnan(seg)
    return float(seg[m].mean()) if m.any() else float("nan")


def _running_slope(y: np.ndarray, up_to_idx: int) -> float:
    seg = y[: up_to_idx + 1]
    m = ~np.isnan(seg)
    if m.sum() < 3:
        return float("nan")
    x = np.arange(seg.size)[m]
    return float(np.polyfit(x, seg[m], 1)[0])


LABELS = ["concise", "productive", "repetitive_reasoning", "repetitive_string"]
ATTACK = {"repetitive_reasoning", "repetitive_string"}
DEFAULT_TAU_PROB = 0.5


def true_label(category: str, is_loop: bool) -> str | None:
    """(category, is_loop) -> 4-way label. A repetitive_reasoning sample that
    never looped is not an attack sample and is dropped (returns None)."""
    if category == "concise_reasoning":
        return "concise"
    if category == "productive_reflection":
        return "productive"
    if category == "repetitive_string":
        return "repetitive_string"
    if "repetitive" in category:
        return "repetitive_reasoning" if is_loop else None
    return None


def load_trajectories(traj_dirs: list[str]) -> list[dict]:
    """Read labelled step_trajectory JSONs into {label, prompt_len, cons, pmf, …}."""
    import glob as _glob

    out: list[dict] = []
    for d in traj_dirs:
        for f in _glob.glob(os.path.join(d, "**", "*.json"), recursive=True):
            if os.path.basename(f) == "manifest.json":
                continue
            try:
                dd = json.load(open(f))
            except Exception:
                continue
            pmf = np.asarray(dd.get("prompt_mean_fraction", []), dtype=float)
            cons = np.asarray(dd.get("attn_consistency_independent", []), dtype=float)
            if pmf.size < MIN_GEN_FOR_SLOPE:
                continue
            lab = true_label(dd.get("category", ""), bool(dd.get("is_loop_sample", False)))
            if lab is None:
                continue
            model = os.path.basename(os.path.dirname(f))
            out.append({
                "file": f, "model": model, "label": lab,
                "prompt_len": int(dd.get("prompt_len", 0)),
                "category": dd.get("category", ""),
                "id": int(dd.get("id", -1)),
                "pmf": pmf, "cons": cons,
                "group_id": f"{model}|{dd.get('category','')}|{dd.get('id',-1)}",
            })
    return out


class Detector:
    """Uniform wrapper over the trained classifier (sklearn pipe or JSON export).

    Attributes mirror the saved bundle: ``classes`` (label order of the
    probability vector), ``use_L`` / ``log_L`` (feature spec), ``feature_names``,
    ``test_group_ids`` (trajectories held out at training time).
    """

    def __init__(self, spec: dict[str, Any], predict_proba):
        self.classes: list[str] = list(spec["classes"])
        self.use_L: bool = bool(spec.get("use_L", True))
        self.log_L: bool = bool(spec.get("log_L", True))
        self.feature_names: list[str] = list(spec.get("feature_names", []))
        self.test_group_ids: list[str] = [str(g) for g in spec.get("test_group_ids", [])]
        self.meta: dict[str, Any] = spec.get("meta", {})
        self.source: str = spec.get("source", "")
        self._predict_proba = predict_proba

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._predict_proba(np.atleast_2d(np.asarray(X, float)))

    def predict_one(self, feat: np.ndarray) -> tuple[str, float, dict[str, float]]:
        p = self.predict_proba(feat)[0]
        j = int(np.argmax(p))
        return self.classes[j], float(p[j]), {c: float(v) for c, v in zip(self.classes, p)}


def _detector_from_joblib(path: str) -> Detector:
    import joblib
    import sklearn

    bundle = joblib.load(path)
    saved = bundle.get("sklearn_version")
    if saved != sklearn.__version__:
        print(f"[detector] WARNING: bundle trained under sklearn {saved}, "
              f"running {sklearn.__version__}", file=sys.stderr)
    pipe = bundle["pipeline"]
    spec = {
        "classes": list(pipe.classes_),
        "use_L": bundle.get("use_L", True),
        "log_L": bundle.get("log_L", True),
        "feature_names": bundle.get("feature_names", []),
        "test_group_ids": bundle.get("test_group_ids", []),
        "source": path,
        "meta": {k: v for k, v in bundle.items()
                 if k not in ("pipeline", "test_group_ids")},
    }
    return Detector(spec, pipe.predict_proba)


def _detector_from_json(path: str) -> Detector:
    blob = json.load(open(path))
    mean = np.asarray(blob["scaler_mean"], float)
    scale = np.asarray(blob["scaler_scale"], float)
    coef = np.asarray(blob["coef"], float)            # (n_classes, n_features)
    intercept = np.asarray(blob["intercept"], float)  # (n_classes,)

    def predict_proba(X: np.ndarray) -> np.ndarray:
        Z = (X - mean) / scale
        logits = Z @ coef.T + intercept
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    blob["source"] = path
    return Detector(blob, predict_proba)


def load_detector(path: str) -> Detector:
    """Load a .joblib bundle (needs sklearn) or its .json export (numpy only)."""
    return (_detector_from_json(path) if path.endswith(".json")
            else _detector_from_joblib(path))


def checkpoint_index(L: int, prompt_len: int, n_gen: int) -> int | None:
    """Probe index into the per-gen-token arrays for total length ``L``.

    Returns None when the sample has not reached ``L`` yet, or when the gen
    segment is still too short for a fittable slope (>=3 points).
    """
    i = L - prompt_len - 1
    if i < MIN_GEN_FOR_SLOPE - 1 or i >= n_gen:
        return None
    return i


def features_at(cons: np.ndarray, pmf: np.ndarray, prompt_len: int, i: int,
                log_L: bool = True, use_L: bool = True) -> np.ndarray | None:
    """Running (cons_ind_mean, pmf_slope[, log L]) over gen positions [0, i]."""
    cm = _running_mean(cons, i)
    sl = _running_slope(pmf, i)
    if np.isnan(cm) or np.isnan(sl):
        return None
    feat = [cm, sl]
    if use_L:
        L = prompt_len + i + 1
        feat.append(float(np.log(L)) if log_L else float(L))
    return np.asarray(feat, dtype=float)


@dataclass
class CheckpointCall:
    """One checkpoint probe and what the detector said there."""
    L: int
    gen_pos: int
    cons_ind_mean: float
    pmf_slope: float
    pred: str
    prob: float
    probs: dict[str, float]
    decided: bool
    is_attack_call: bool


@dataclass
class DetectionTrace:
    """The judgement trajectory of one sample (design §2.2 step 4)."""
    calls: list[CheckpointCall] = field(default_factory=list)

    def add(self, det: Detector, L: int, i: int, cons: np.ndarray,
            pmf: np.ndarray, prompt_len: int,
            tau_prob: float = DEFAULT_TAU_PROB) -> CheckpointCall | None:
        feat = features_at(cons, pmf, prompt_len, i,
                           log_L=det.log_L, use_L=det.use_L)
        if feat is None:
            return None
        pred, prob, probs = det.predict_one(feat)
        decided = prob >= tau_prob
        call = CheckpointCall(
            L=L, gen_pos=i, cons_ind_mean=float(feat[0]), pmf_slope=float(feat[1]),
            pred=pred, prob=prob, probs=probs, decided=decided,
            is_attack_call=bool(decided and pred in ATTACK),
        )
        self.calls.append(call)
        return call

    # --- trajectory-level policy -------------------------------------------
    @property
    def trigger(self) -> CheckpointCall | None:
        """First confident attack call — the online stop signal."""
        for c in self.calls:
            if c.is_attack_call:
                return c
        return None

    @property
    def verdict(self) -> str | None:
        """Deployed trajectory verdict: the trigger's class if one fired,
        else the last decided (benign) class, else None (never decided)."""
        t = self.trigger
        if t is not None:
            return t.pred
        for c in reversed(self.calls):
            if c.decided:
                return c.pred
        return None

    def as_dict(self) -> dict[str, Any]:
        t = self.trigger
        return {
            "calls": [vars(c) for c in self.calls],
            "trigger_L": t.L if t else None,
            "trigger_pred": t.pred if t else None,
            "verdict": self.verdict,
        }

    @classmethod
    def from_dict(cls, blob: dict[str, Any]) -> "DetectionTrace":
        return cls(calls=[CheckpointCall(**c) for c in blob.get("calls", [])])


def replay_trace(det: Detector, cons: np.ndarray, pmf: np.ndarray,
                 prompt_len: int, tau_prob: float = DEFAULT_TAU_PROB,
                 checkpoints: list[int] | None = None) -> DetectionTrace:
    """Run the full checkpoint protocol over already-computed metric arrays."""
    trace = DetectionTrace()
    n_gen = int(pmf.size)
    for L in (checkpoints or CHECKPOINTS):
        i = checkpoint_index(L, prompt_len, n_gen)
        if i is None:
            continue
        trace.add(det, L, i, cons, pmf, prompt_len, tau_prob)
    return trace


def _export(bundle_path: str, out_path: str) -> None:
    """joblib bundle -> sklearn-free JSON, verified against sklearn."""
    import joblib
    import sklearn

    bundle = joblib.load(bundle_path)
    pipe = bundle["pipeline"]
    scaler = pipe.named_steps["standardscaler"]
    logit = pipe.named_steps["logisticregression"]
    blob = {
        "classes": [str(c) for c in logit.classes_],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "coef": logit.coef_.tolist(),
        "intercept": logit.intercept_.tolist(),
        "use_L": bool(bundle.get("use_L", True)),
        "log_L": bool(bundle.get("log_L", True)),
        "feature_names": list(bundle.get("feature_names", [])),
        "test_group_ids": [str(g) for g in bundle.get("test_group_ids", [])],
        "meta": {"exported_from": bundle_path,
                 "sklearn_version": sklearn.__version__,
                 "n_train_rows": bundle.get("n_train_rows"),
                 "n_train_trajectories": bundle.get("n_train_trajectories"),
                 "test_frac": bundle.get("test_frac"),
                 "seed": bundle.get("seed")},
    }
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=1)

    # Fidelity check: the numpy path must reproduce sklearn's probabilities.
    rng = np.random.default_rng(0)
    n_feat = len(blob["scaler_mean"])
    X = np.column_stack([
        rng.uniform(-0.2, 1.0, 2000),                       # cons_ind
        rng.uniform(-0.02, 0.05, 2000),                     # pmf_slope
    ] + ([rng.uniform(np.log(64), np.log(16384), 2000)] if n_feat == 3 else []))
    ref = pipe.predict_proba(X)
    got = _detector_from_json(out_path).predict_proba(X)
    dmax = float(np.abs(ref - got).max())
    print(f"[detector] exported {bundle_path} -> {out_path}")
    print(f"[detector] numpy vs sklearn max|Δp| = {dmax:.3e} over {len(X)} random probes")
    if dmax > 1e-9:
        raise SystemExit("[detector] JSON export does not reproduce sklearn — abort")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--export", required=True, help="output .json path")
    a = ap.parse_args()
    _export(a.bundle, a.export)
