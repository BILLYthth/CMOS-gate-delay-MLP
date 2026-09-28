import os
import math
import warnings
import joblib
import numpy as np

warnings.filterwarnings("ignore")

# ==========================================================
# PATH
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_FILES = {
    "NAND": "NAND_final_model.pkl",
    "NOR": "NOR_final_model.pkl",
    "EXOR": "EXOR_final_model.pkl"
}

_loaded_models = {}

# ==========================================================
# LOAD MODEL
# ==========================================================
# NOTE: unlike the reference script, this bundle stores a single sklearn
# Pipeline (scaler + mlp already chained together) under "model", plus the
# feature order under "feature_cols". There is no separate obj["scaler"] --
# the pipeline scales internally, so build_feature_vector() below does NOT
# call .transform() itself; predict_delay() just calls pipeline.predict(X)
# directly on the raw (unscaled) feature vector.

def load_model(gate):

    gate = gate.strip().upper()

    if gate in _loaded_models:
        return _loaded_models[gate]

    path = os.path.join(MODEL_DIR, MODEL_FILES[gate])

    obj = joblib.load(path)

    _loaded_models[gate] = obj

    return obj


# ==========================================================
# DEFAULT Req
# ==========================================================

def get_default_req(obj):

    features = obj["feature_cols"]

    if "log_Req" not in features:
        return None

    idx = features.index("log_Req")

    scaler = obj["model"].named_steps["scaler"]

    mean_log = float(scaler.mean_[idx])

    return 10**mean_log


# ==========================================================
# BUILD FEATURE VECTOR
# ==========================================================

def build_feature_vector(
        obj,
        Wn,
        Wp,
        CL,
        VDD,
        Temp,
        pin):

    K = Wp / Wn

    pin_enc = 1 if pin.upper() == "A" else 0

    feature_dict = {

        "VDD": VDD,

        "log_CL": math.log10(CL),

        "log_Wn": math.log10(Wn),

        "K": K,

        "Temp": Temp,

        "log_Wp": math.log10(Wp),

        "pin_enc": pin_enc

    }

    # -----------------------------------------
    # Only add Req if this model needs it
    # -----------------------------------------

    features = obj["feature_cols"]

    if "log_Req" in features:

        Req = get_default_req(obj)

        feature_dict["log_Req"] = math.log10(Req)

    # -----------------------------------------
    # Build feature vector in correct order
    # -----------------------------------------

    X = []

    for feat in features:

        if feat not in feature_dict:

            raise ValueError(f"Missing feature: {feat}")

        X.append(feature_dict[feat])

    return np.array(X).reshape(1, -1)


# ==========================================================
# PREDICT
# ==========================================================

def predict_delay(
        gate,
        Wn,
        Wp,
        CL,
        VDD,
        Temp,
        pin):

    obj = load_model(gate)

    X = build_feature_vector(
        obj,
        Wn,
        Wp,
        CL,
        VDD,
        Temp,
        pin
    )

    # Pipeline handles scaling internally -- feed it the raw feature vector.
    log_tp = obj["model"].predict(X)[0]

    Tp = 10**log_tp

    return Tp


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("Propagation Delay Predictor")
    print("=" * 60)

    gate = input("Gate (NAND/NOR/EXOR): ").strip().upper()

    Wn_nm = float(input("Wn (nm): "))
    Wp_nm = float(input("Wp (nm): "))
    VDD = float(input("VDD (V): "))
    CL_fF = float(input("CL (fF): "))
    Temp = float(input("Temperature (C): "))

    # Unit conversion

    Wn = Wn_nm * 1e-9
    Wp = Wp_nm * 1e-9
    CL = CL_fF * 1e-15

    Tp_A = predict_delay(
        gate,
        Wn,
        Wp,
        CL,
        VDD,
        Temp,
        "A"
    )

    Tp_B = predict_delay(
        gate,
        Wn,
        Wp,
        CL,
        VDD,
        Temp,
        "B"
    )

    print()

    print("=" * 50)

    print("Prediction Result")

    print("=" * 50)

    print(f"Gate      : {gate}")

    print(f"Tp Pin A  : {Tp_A*1e12:.3f} ps")

    print(f"Tp Pin B  : {Tp_B*1e12:.3f} ps")

    print(f"Worst Tp  : {max(Tp_A, Tp_B)*1e12:.3f} ps")

    print("=" * 50)