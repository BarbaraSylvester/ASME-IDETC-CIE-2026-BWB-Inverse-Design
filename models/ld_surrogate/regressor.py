#!/usr/bin/env python3
"""Direct surrogate:  (geom[9] + flight[3]) -> (CL, CD).

PURE NUMPY implementation (Adam-trained MLP). Deliberately torch-free: in the
neuralsolver env on CPU nodes, `import torch` times out (>120s) and sklearn/pyg
do too, while numpy+scipy import instantly. Since the optimization objective only
needs this small regressor, keeping it numpy-only makes the whole core (regressor
+ objective + baselines + scorers) fast and reliable on mit_normal/mit_preemptable.
(The heavyweight FIELD surrogates that genuinely need torch live in fields.py and
are an optional, GPU-partition concern.)

Trained on the CSV's CFD ground-truth CL/CD. Decision: TRAIN on FULL data (more
accurate), restrict the optimizer's SEARCH BOX to feasible bounds + penalty.

    $NS_PY surrogate/regressor.py --data full --epochs 4000
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

ROOT = os.environ.get("BNPP_OPT_ROOT", "/orcd/data/faez/001/nick/bnpp_optimize")


# --------------------------- pure-numpy MLP --------------------------------
def _he_init(rng, fan_in, fan_out):
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


class NumpyMLP:
    """SiLU MLP with Adam. Weights stored as list of (W, b)."""

    def __init__(self, in_dim=12, hid=128, out_dim=2, depth=3, seed=0):
        rng = np.random.default_rng(seed)
        dims = [in_dim] + [hid] * depth + [out_dim]
        self.W = [_he_init(rng, dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        self.b = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
        self.dims = dims

    @staticmethod
    def _silu(x):
        return x / (1.0 + np.exp(-x))

    @staticmethod
    def _silu_grad(x):
        s = 1.0 / (1.0 + np.exp(-x))
        return s * (1.0 + x * (1.0 - s))

    def forward(self, X, cache=False):
        a = X; pre = []; act = [a]
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            pre.append(z)
            if i < len(self.W) - 1:
                a = self._silu(z)
            else:
                a = z  # linear output
            act.append(a)
        if cache:
            return a, pre, act
        return a

    def backward(self, pre, act, dout):
        gW = [None] * len(self.W); gb = [None] * len(self.W)
        d = dout
        for i in reversed(range(len(self.W))):
            gW[i] = act[i].T @ d / d.shape[0]
            gb[i] = d.mean(0)
            if i > 0:
                d = (d @ self.W[i].T) * self._silu_grad(pre[i - 1])
        return gW, gb

    def params(self):
        return self.W + self.b

    def state(self):
        return {"W": [w.tolist() for w in self.W], "b": [bb.tolist() for bb in self.b],
                "dims": self.dims}

    @classmethod
    def from_state(cls, st):
        m = cls.__new__(cls)
        m.W = [np.asarray(w) for w in st["W"]]
        m.b = [np.asarray(bb) for bb in st["b"]]
        m.dims = st["dims"]
        return m


class Standardizer:
    def __init__(self, xm, xs, ym, ys):
        self.xm, self.xs, self.ym, self.ys = (np.asarray(v, dtype=np.float64)
                                              for v in (xm, xs, ym, ys))

    def to_json(self):
        return {k: getattr(self, k).tolist() for k in ["xm", "xs", "ym", "ys"]}

    @classmethod
    def from_json(cls, d):
        return cls(d["xm"], d["xs"], d["ym"], d["ys"])


def _load(data_tag):
    d = np.load(os.path.join(ROOT, "data", f"regressor_data_{data_tag}.npz"), allow_pickle=True)
    X = np.concatenate([d["geom"], d["flight"]], axis=1).astype(np.float64)
    Y = np.stack([d["CL"], d["CD"]], axis=1).astype(np.float64)
    return X, Y


def train(data_tag="full", epochs=4000, hid=128, depth=3, lr=2e-3,
          val_frac=0.15, seed=0, batch=1024):
    np.random.seed(seed)
    X, Y = _load(data_tag)
    n = len(X); idx = np.random.permutation(n); nval = int(n * val_frac)
    vva, vtr = idx[:nval], idx[nval:]
    Xtr, Ytr, Xva, Yva = X[vtr], Y[vtr], X[vva], Y[vva]
    xm, xs = Xtr.mean(0), Xtr.std(0) + 1e-8
    ym, ys = Ytr.mean(0), Ytr.std(0) + 1e-8
    std = Standardizer(xm, xs, ym, ys)
    Xt = (Xtr - xm) / xs; Yt = (Ytr - ym) / ys
    Xv = (Xva - xm) / xs; Yv = (Yva - ym) / ys

    model = NumpyMLP(in_dim=X.shape[1], hid=hid, out_dim=2, depth=depth, seed=seed)
    P = model.params()
    mt = [np.zeros_like(p) for p in P]; vt = [np.zeros_like(p) for p in P]
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(seed)
    nW = len(model.W)
    best = (1e18, None); t = 0
    steps_per = max(1, len(Xt) // batch)
    for ep in range(epochs):
        perm = rng.permutation(len(Xt))
        for s in range(steps_per):
            bi = perm[s * batch:(s + 1) * batch]
            xb, yb = Xt[bi], Yt[bi]
            pred, pre, act = model.forward(xb, cache=True)
            dout = 2.0 * (pred - yb) / yb.shape[0]
            gW, gb = model.backward(pre, act, dout)
            grads = gW + gb; t += 1
            for j, g in enumerate(grads):
                mt[j] = b1 * mt[j] + (1 - b1) * g
                vt[j] = b2 * vt[j] + (1 - b2) * (g * g)
                mhat = mt[j] / (1 - b1 ** t); vhat = vt[j] / (1 - b2 ** t)
                upd = lr * mhat / (np.sqrt(vhat) + eps)
                if j < nW:
                    model.W[j] -= upd
                else:
                    model.b[j - nW] -= upd
        if ep % 50 == 0 or ep == epochs - 1:
            vl = float(((model.forward(Xv) - Yv) ** 2).mean())
            if vl < best[0]:
                best = (vl, (
                    [w.copy() for w in model.W], [bb.copy() for bb in model.b]))
    if best[1] is not None:
        model.W, model.b = best[1]

    pred = model.forward(Xv) * ys + ym
    true = Yva; metrics = {}
    for j, name in enumerate(["CL", "CD"]):
        ti, pi = true[:, j], pred[:, j]
        ss = float(((ti - pi) ** 2).sum()); st = float(((ti - ti.mean()) ** 2).sum())
        metrics[name] = {"R2": 1 - ss / st, "MAE": float(np.abs(ti - pi).mean())}
    lt = true[:, 0] / np.clip(true[:, 1], 1e-6, None)
    lp = pred[:, 0] / np.clip(pred[:, 1], 1e-6, None)
    ss = float(((lt - lp) ** 2).sum()); st = float(((lt - lt.mean()) ** 2).sum())
    metrics["LoD"] = {"R2": 1 - ss / st, "MAE": float(np.abs(lt - lp).mean())}

    os.makedirs(os.path.join(ROOT, "regressor_ckpt"), exist_ok=True)
    ck = os.path.join(ROOT, "regressor_ckpt", f"reg_{data_tag}.json")
    json.dump({"model": model.state(), "std": std.to_json(),
               "data_tag": data_tag, "metrics": metrics, "n": n},
              open(ck, "w"))
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    json.dump({"ckpt": ck, "n": n, "metrics": metrics},
              open(os.path.join(ROOT, "logs", f"reg_{data_tag}.json"), "w"), indent=2)
    print(json.dumps({"data": data_tag, "n": n, "metrics": metrics}, indent=2))
    return ck, metrics


# Column order is NOT stored in the checkpoint -- it is fixed at data-prep time.
# Geometry is in mm at a constant root chord C1 = 1000 mm; C1 reaches the model
# only through Re_L (reference length = root chord, in metres).
GEOM_FEATURES = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S3", "X3"]
FLIGHT_FEATURES = ["Re_L", "M_inf", "alpha_deg"]

_CACHE = {}


def load(data_tag="full", device="cpu"):
    if data_tag in _CACHE:
        return _CACHE[data_tag]
    # Vendored copy: checkpoints sit beside this file; fall back to the source repo.
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"reg_{data_tag}.json")
    ck = json.load(open(local if os.path.exists(local)
                        else os.path.join(ROOT, "regressor_ckpt", f"reg_{data_tag}.json")))
    m = NumpyMLP.from_state(ck["model"]); std = Standardizer.from_json(ck["std"])
    _CACHE[data_tag] = (m, std)
    return m, std


def predict(model, std, geom, flight, device="cpu"):
    X = np.concatenate([np.atleast_2d(geom), np.atleast_2d(flight)], axis=1).astype(np.float64)
    Xn = (X - std.xm) / std.xs
    out = model.forward(Xn)
    return out * std.ys + std.ym  # (n,2) = [CL, CD]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["feasible", "full"], default="full")
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--hid", type=int, default=128)
    ap.add_argument("--depth", type=int, default=3)
    a = ap.parse_args()
    train(a.data, epochs=a.epochs, hid=a.hid, depth=a.depth)
