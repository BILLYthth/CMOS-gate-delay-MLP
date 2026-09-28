"""
Fast propagation-delay predictor -- NO scikit-learn / joblib import.

Why this is faster:
    Importing scikit-learn pulls in scipy and a large dependency chain --
    on a typical machine that alone costs 1-3 seconds, every single run,
    even though the actual MLP forward pass (a handful of matrix
    multiplications) takes microseconds. This script loads the trained
    weights from lightweight .npz files (produced once by
    export_weights.py) and does the forward pass with plain NumPy, so
    startup + prediction together are near-instant.

Setup (one-time, needs scikit-learn installed):
    python export_weights.py
    -> creates weights/NAND_weights.npz, NOR_weights.npz, EXOR_weights.npz

Then for every actual run, only this script (and numpy) is needed:
    python predict_fast.py
"""

import os
import math
import time
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

MODEL_FILES = {
    "NAND": "NAND_weights.npz",
    "NOR": "NOR_weights.npz",
    "EXOR": "EXOR_weights.npz",
}

_loaded = {}


def load_weights(gate):
    gate = gate.strip().upper()
    if gate in _loaded:
        return _loaded[gate]

    path = os.path.join(WEIGHTS_DIR, MODEL_FILES[gate])
    data = np.load(path, allow_pickle=False)

    n_layers = int(data["n_layers"])
    weights = {
        "feature_cols": [str(f) for f in data["feature_cols"]],
        "scaler_mean": data["scaler_mean"],
        "scaler_scale": data["scaler_scale"],
        "activation": str(data["activation"]),
        "n_layers": n_layers,
        "W": [data[f"W{i}"] for i in range(n_layers)],
        "b": [data[f"b{i}"] for i in range(n_layers)],
    }
    _loaded[gate] = weights
    return weights


def _activate(z, kind):
    if kind == "tanh":
        return np.tanh(z)
    if kind == "relu":
        return np.maximum(z, 0)
    if kind == "logistic":
        return 1.0 / (1.0 + np.exp(-z))
    if kind == "identity":
        return z
    raise ValueError(f"Unsupported activation: {kind}")


def mlp_forward(weights, x_scaled):
    """Manual forward pass replicating sklearn's MLPRegressor.predict()."""
    a = x_scaled
    n_layers = weights["n_layers"]
    for i in range(n_layers):
        z = a @ weights["W"][i] + weights["b"][i]
        # hidden layers use the trained activation; output layer is linear
        a = _activate(z, weights["activation"]) if i < n_layers - 1 else z
    return a


def get_default_req(weights):
    features = weights["feature_cols"]
    if "log_Req" not in features:
        return None
    idx = features.index("log_Req")
    mean_log = float(weights["scaler_mean"][idx])
    return 10**mean_log


def build_feature_vector(weights, Wn, Wp, CL, VDD, Temp, pin):
    K = Wp / Wn
    pin_enc = 1 if pin.upper() == "A" else 0

    feature_dict = {
        "VDD": VDD,
        "log_CL": math.log10(CL),
        "log_Wn": math.log10(Wn),
        "K": K,
        "Temp": Temp,
        "log_Wp": math.log10(Wp),
        "pin_enc": pin_enc,
    }

    features = weights["feature_cols"]
    if "log_Req" in features:
        Req = get_default_req(weights)
        feature_dict["log_Req"] = math.log10(Req)

    X = np.array([[feature_dict[f] for f in features]], dtype=np.float64)
    return X


def predict_delay(gate, Wn, Wp, CL, VDD, Temp, pin):
    weights = load_weights(gate)
    X = build_feature_vector(weights, Wn, Wp, CL, VDD, Temp, pin)

    # manual StandardScaler transform
    X_scaled = (X - weights["scaler_mean"]) / weights["scaler_scale"]

    log_tp = mlp_forward(weights, X_scaled)[0, 0]
    Tp = 10**log_tp
    return Tp


if __name__ == "__main__":

    print("=" * 60)
    print("Propagation Delay Predictor (fast, numpy-only)")
    print("=" * 60)

    gate = input("Gate (NAND/NOR/EXOR): ").strip().upper()

    Wn_nm = float(input("Wn (nm): "))
    Wp_nm = float(input("Wp (nm): "))
    VDD = float(input("VDD (V): "))
    CL_fF = float(input("CL (fF): "))
    Temp = float(input("Temperature (C): "))

    Wn = Wn_nm * 1e-9
    Wp = Wp_nm * 1e-9
    CL = CL_fF * 1e-15

    # Load weights once (this is the only step that touches disk); time it
    # separately from inference so the per-pin numbers below reflect pure
    # forward-pass cost, not file I/O.
    t_load0 = time.perf_counter()
    load_weights(gate)
    t_load1 = time.perf_counter()
    load_ms = (t_load1 - t_load0) * 1000

    t0 = time.perf_counter()
    Tp_A = predict_delay(gate, Wn, Wp, CL, VDD, Temp, "A")
    t1 = time.perf_counter()
    Tp_B = predict_delay(gate, Wn, Wp, CL, VDD, Temp, "B")
    t2 = time.perf_counter()

    pin_a_ms = (t1 - t0) * 1000
    pin_b_ms = (t2 - t1) * 1000
    total_ms = load_ms + pin_a_ms + pin_b_ms

    print()
    print("=" * 50)