import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename

# Optional: torch may be needed depending on how the models were saved
try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None

from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EEG_EXT = {".npy"}
ALLOWED_AUDIO_EXT = {".wav"}

APP = Flask(__name__)
APP.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Paths to provided artifacts (assumed present in repo root)
MODEL_OPENSMILE_PATH = os.path.join(BASE_DIR, "GESSNet_Opensmile_3.pkl")
MODEL_YAMNET_PATH = os.path.join(BASE_DIR, "GESSNet_Yamnet_4.pkl")
OPENSMILE_CSV = os.path.join(BASE_DIR, "OpenSMILE_features.csv")
YAMNET_CSV = os.path.join(BASE_DIR, "YAMNet_features.csv")


def load_model(path):
    """Attempt to load a pickled or torch-saved model."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # try pickle
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model
    except Exception:
        pass

    # try torch
    if torch is not None:
        try:
            loaded = torch.load(path, map_location="cpu")
            # If the file is a state_dict (OrderedDict), try to locate a local
            # `model_arch.py` that provides a `get_model(name=None)` factory.
            if isinstance(loaded, dict):
                # attempt to auto-build model from model_arch.py
                arch_path = os.path.join(BASE_DIR, "model_arch.py")
                if os.path.exists(arch_path):
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("model_arch", arch_path)
                    mod = importlib.util.module_from_spec(spec)
                    try:
                        spec.loader.exec_module(mod)
                        # prefer specific factory functions if available
                        name = os.path.splitext(os.path.basename(path))[0]
                        model = None
                        # try to infer song_feature_dim, num_classes, and input_channels from state_dict
                        song_feature_dim = None
                        num_classes = None
                        input_channels = None
                        try:
                            # song_project.*.weight typically has shape (512, song_feature_dim) or similar
                            for k, v in loaded.items():
                                if 'song_project.0.weight' in k or 'song_attention.0.weight' in k:
                                    if hasattr(v, 'shape') and len(v.shape) >= 2:
                                        song_feature_dim = v.shape[1]
                                if 'fc_combined' in k and getattr(v, 'ndim', 0) == 2:
                                    num_classes = v.shape[0]
                                if 'depthwiseConv.0.weight' in k and hasattr(v, 'shape'):
                                    # weight shape is (out_channels, 1, input_channels, 1)
                                    if len(v.shape) >= 3:
                                        input_channels = v.shape[2]
                            # fallback heuristics
                            if song_feature_dim is None:
                                for k, v in loaded.items():
                                    if 'song_project' in k and getattr(v, 'ndim', 0) == 2:
                                        song_feature_dim = v.shape[1]
                                        break
                            if num_classes is None:
                                for k, v in loaded.items():
                                    if 'fc_combined' in k and getattr(v, 'ndim', 0) == 2:
                                        num_classes = v.shape[0]
                                        break
                        except Exception:
                            pass

                        get_model = getattr(mod, "get_model", None)
                        if get_model is not None:
                            try:
                                model = get_model(name)
                            except TypeError:
                                # allow passing inferred dims
                                try:
                                    model = get_model(name, song_feature_dim=song_feature_dim, num_classes=num_classes, input_channels=input_channels)
                                except Exception:
                                    try:
                                        model = get_model(song_feature_dim=song_feature_dim, num_classes=num_classes, input_channels=input_channels)
                                    except Exception:
                                        model = get_model()
                        else:
                            base = os.path.basename(path).lower()
                            # attempt model-specific factories following your suggestion
                            if "opensmile" in base:
                                gm = getattr(mod, "get_model_opensmile", None)
                            elif "yamnet" in base or "yam" in base:
                                gm = getattr(mod, "get_model_yamnet", None)
                            else:
                                gm = None

                            if gm is not None:
                                try:
                                    model = gm(name)
                                except TypeError:
                                    try:
                                        model = gm(name, song_feature_dim=song_feature_dim, num_classes=num_classes, input_channels=input_channels)
                                    except Exception:
                                        try:
                                            model = gm(song_feature_dim=song_feature_dim, num_classes=num_classes, input_channels=input_channels)
                                        except Exception:
                                            model = gm()

                        if model is not None:
                            try:
                                # user-provided model instance should accept a state_dict
                                model.load_state_dict(loaded)
                                model.eval()
                                return model
                            except Exception:
                                # if loading fails, fall back to returning state_dict
                                pass
                    except Exception:
                        # if model_arch fails to import/execute, continue and return the raw state_dict
                        pass
                # return the mapping if we couldn't instantiate model
                return loaded
            return loaded
        except Exception as e:
            # Some torch checkpoints (full model pickles) require allowing
            # custom globals to be unpickled. Try a safer retry path where we
            # import `model_arch.py`, register its classes with torch's
            # safe globals, inject them into __main__, and re-run torch.load
            # with weights_only=False. This is only attempted when a
            # model_arch exists locally and torch exposes the required APIs.
            # If the filename suggests these are the newer pickles (user-provided),
            # try the 'safe globals' approach using classes from model_arch.py.
            try:
                if '_new' in os.path.basename(path):
                    arch_path = os.path.join(BASE_DIR, "model_arch.py")
                    if os.path.exists(arch_path):
                        import importlib.util, sys, inspect
                        spec = importlib.util.spec_from_file_location("model_arch", arch_path)
                        mod = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(mod)
                        # try to find classes with expected names
                        candidate_names = ["GESSNetWithOpenSMILE", "GESSNetWithYAMNet", "GESSNet", "GESSNetModel"]
                        classes = []
                        for cname in candidate_names:
                            if hasattr(mod, cname):
                                cls = getattr(mod, cname)
                                classes.append(cls)

                        # also collect any classes present as fallback
                        if not classes:
                            for name in dir(mod):
                                obj = getattr(mod, name)
                                if inspect.isclass(obj):
                                    classes.append(obj)

                        # register safe globals if API available
                        reg = getattr(torch.serialization, 'add_safe_globals', None)
                        if reg is not None and classes:
                            try:
                                reg(classes)
                            except Exception:
                                pass

                        # inject classes into __main__ so pickle can find them by name
                        try:
                            import __main__
                            for cls in classes:
                                setattr(__main__, cls.__name__, cls)
                        except Exception:
                            pass

                        # retry loading with weights_only=False (user-suggested)
                        try:
                            loaded = torch.load(path, map_location="cpu", weights_only=False)
                        except TypeError:
                            loaded = torch.load(path, map_location="cpu")
                        return loaded
            except Exception:
                pass

    raise RuntimeError(f"Could not load model at {path}")


def load_song_features(csv_path):
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path, index_col=0)
    # normalize and drop very low-variance features
    scaler = StandardScaler()
    arr = df.values.astype(np.float32)
    arr = scaler.fit_transform(arr)
    df = pd.DataFrame(arr, index=df.index, columns=df.columns)
    var = df.var()
    df = df.loc[:, var > 0.01]
    return df.to_dict(orient="index")


def allowed_file(filename, allowed_exts):
    ext = os.path.splitext(filename)[1].lower()
    return ext in allowed_exts


def prepare_eeg(eeg_np):
    eeg = eeg_np.astype(np.float32)
    if eeg.ndim != 2:
        # try to coerce to (channels, time)
        eeg = eeg.reshape(eeg.shape[0], -1).astype(np.float32)
    if eeg.shape[1] > 100:
        eeg = eeg - np.mean(eeg, axis=1, keepdims=True).astype(np.float32)
    return eeg


def predict_with_model(model, eeg, song_feat):
    """Heuristic wrapper to call different model types.

    Returns a dict with at least a 'pred' key and optional 'probs'.
    """
    # If the loaded object is a mapping (e.g. a state_dict / OrderedDict),
    # it isn't callable — return a helpful error instead of attempting to call it.
    if isinstance(model, dict):
        return {"error": "Loaded model is a mapping (e.g. state_dict / OrderedDict).\n" \
                           "Provide a callable model object (pickle the model itself) or load the state_dict into the model class before running predictions."}
    # sklearn-like estimator
    try:
        if hasattr(model, "predict_proba") or hasattr(model, "predict"):
            flat = eeg.flatten()
            X = np.concatenate([flat, song_feat]).reshape(1, -1)
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X)
                # classes_ may exist
                classes = getattr(model, "classes_", None)
                if classes is not None:
                    pred = classes[np.argmax(probs, axis=1)][0]
                else:
                    pred = int(np.argmax(probs, axis=1)[0])
                return {"pred": str(pred), "probs": probs.tolist()}
            else:
                pred = model.predict(X)
                return {"pred": str(pred[0])}
    except Exception:
        pass

    # torch.nn.Module
    if torch is not None and isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            eeg_t = torch.tensor(eeg, dtype=torch.float32).unsqueeze(0)  # batch
            song_t = torch.tensor(song_feat, dtype=torch.float32).unsqueeze(0)
            try:
                out = model(eeg_t, song_t)
            except TypeError:
                # try concatenation fallback
                inp = torch.cat([eeg_t.flatten(1), song_t], dim=1)
                out = model(inp)

            if isinstance(out, torch.Tensor):
                if out.ndim == 2:
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                    pred_idx = int(probs.argmax(axis=1)[0])
                    return {"pred": pred_idx, "probs": probs.tolist()}
                else:
                    return {"pred": out.cpu().numpy().tolist()}

    # fallback: try calling directly
    try:
        out = model(eeg, song_feat)
        return {"pred": str(out)}
    except Exception as e:
        return {"error": str(e)}


# mapping from model class index to label (use exactly as provided)
CLASS_MAP = {
    0: 'HDHV',
    1: 'LDHV',
    2: 'LDLV',
    3: 'HDLV'
}


def normalize_prediction(res):
    """Normalize raw model output dict into a stable shape and map class index to label.

    res: dict possibly containing 'pred' (int or str) and/or 'probs'/'probs' list.
    Returns a dict with at least 'pred' (int) and 'label' (mapped string) and 'probabilities' (list) unless error.
    """
    if not isinstance(res, dict):
        return {"error": "invalid prediction result"}
    if 'error' in res:
        return res

    probs = None
    # accept multiple possible keys
    if 'probs' in res:
        probs = res.get('probs')
    elif 'probs' in res:
        probs = res.get('probs')
    elif 'probabilities' in res:
        probs = res.get('probabilities')
    elif 'probs_list' in res:
        probs = res.get('probs_list')

    pred = res.get('pred')
    try:
        if pred is None and probs is not None:
            import numpy as _np
            arr = _np.array(probs)
            if arr.ndim == 2:
                # shape (1, C)
                idx = int(_np.argmax(arr, axis=1)[0])
            else:
                idx = int(_np.argmax(arr))
            pred_idx = idx
        else:
            # try to coerce pred to int
            pred_idx = int(pred)
    except Exception:
        return {"error": "could not determine prediction index"}

    label = CLASS_MAP.get(pred_idx, str(pred_idx))
    out = {
        'pred': int(pred_idx),
        'label': label,
        'probabilities': list(probs) if probs is not None else None,
    }
    # keep compatible key for older frontends
    if out['probabilities'] is not None:
        # flatten if shape is nested like [[...]] -> take first row
        try:
            if len(out['probabilities']) == 1 and isinstance(out['probabilities'][0], (list, tuple)):
                out['probabilities'] = list(out['probabilities'][0])
        except Exception:
            pass
        out['probs'] = out['probabilities']
    return out


# Load models and feature dicts at startup
try:
    MODEL_OPENSMILE = load_model(MODEL_OPENSMILE_PATH)
except Exception as e:
    MODEL_OPENSMILE = None
    print(f"Warning: could not load OpenSMILE model: {e}")

try:
    MODEL_YAMNET = load_model(MODEL_YAMNET_PATH)
except Exception as e:
    MODEL_YAMNET = None
    print(f"Warning: could not load YAMNet model: {e}")

OPENSMILE_FEATURES = load_song_features(OPENSMILE_CSV)
YAMNET_FEATURES = load_song_features(YAMNET_CSV)


@APP.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@APP.route("/predict", methods=["POST"])
def predict():
    global MODEL_OPENSMILE, MODEL_YAMNET
    # files: accept either 'song_file' (new) or 'audio_file' (legacy from older frontend)
    if "eeg_file" not in request.files or ("song_file" not in request.files and "audio_file" not in request.files):
        return jsonify({"error": "Both eeg_file and song_file/audio_file required"}), 400

    eeg_file = request.files.get("eeg_file")
    # prefer 'song_file' but fall back to legacy 'audio_file'
    song_file = request.files.get("song_file") if "song_file" in request.files else request.files.get("audio_file")

    if eeg_file.filename == "" or song_file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(eeg_file.filename, ALLOWED_EEG_EXT) or not allowed_file(song_file.filename, ALLOWED_AUDIO_EXT):
        return jsonify({"error": "Invalid file extension"}), 400

    eeg_fname = secure_filename(eeg_file.filename)
    song_fname = secure_filename(song_file.filename)
    eeg_path = os.path.join(UPLOAD_FOLDER, eeg_fname)
    song_path = os.path.join(UPLOAD_FOLDER, song_fname)
    eeg_file.save(eeg_path)
    song_file.save(song_path)

    # load eeg
    try:
        eeg_np = np.load(eeg_path)
    except Exception as e:
        return jsonify({"error": f"Could not load EEG .npy: {e}"}), 400

    eeg_pre = prepare_eeg(eeg_np)

    # derive song id from filename (filename without extension)
    song_id = os.path.splitext(song_fname)[0]

    results = {}

    # OpenSMILE model prediction
    def _ensure_model_is_callable(model_obj, state_dict_obj, base_name, song_feat):
        """If model_obj is a mapping (state_dict), try to instantiate the architecture and load weights.

        Returns a callable model or the original state_dict if instantiation fails.
        """
        if not isinstance(model_obj, dict):
            return model_obj

        # try import model_arch and use factory
        arch_path = os.path.join(BASE_DIR, "model_arch.py")
        if not os.path.exists(arch_path):
            return model_obj

        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("model_arch", arch_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            # infer dims
            song_feature_dim = int(song_feat.shape[0]) if song_feat is not None else None
            num_classes = None
            input_channels = None
            for k, v in state_dict_obj.items():
                if 'fc_combined' in k and getattr(v, 'ndim', 0) == 2:
                    num_classes = v.shape[0]
                if 'depthwiseConv.0.weight' in k and hasattr(v, 'shape'):
                    # expected shape (out, 1, input_channels, 1)
                    if len(v.shape) >= 3:
                        input_channels = v.shape[2]

            base = base_name.lower()
            factory = None
            if hasattr(mod, 'get_model'):
                factory = getattr(mod, 'get_model')
            elif 'opensmile' in base and hasattr(mod, 'get_model_opensmile'):
                factory = getattr(mod, 'get_model_opensmile')
            elif ('yamnet' in base or 'yam' in base) and hasattr(mod, 'get_model_yamnet'):
                factory = getattr(mod, 'get_model_yamnet')

            if factory is None:
                return model_obj

            # try to instantiate with inferred params
            try:
                m = factory(base, song_feature_dim=song_feature_dim, num_classes=num_classes, input_channels=input_channels)
            except TypeError:
                try:
                    m = factory(song_feature_dim=song_feature_dim, num_classes=num_classes, input_channels=input_channels)
                except Exception:
                    try:
                        m = factory()
                    except Exception:
                        return model_obj

            try:
                m.load_state_dict(state_dict_obj)
                m.eval()
                return m
            except Exception:
                return model_obj
        except Exception:
            return model_obj

    # OpenSMILE model prediction
    if MODEL_OPENSMILE is not None:
        if song_id in OPENSMILE_FEATURES:
            song_feat = np.array(list(OPENSMILE_FEATURES[song_id].values()), dtype=np.float32)
            # if loaded artifact is a state_dict, try to build model now using song_feat
            MODEL_OPENSMILE = _ensure_model_is_callable(MODEL_OPENSMILE, MODEL_OPENSMILE if isinstance(MODEL_OPENSMILE, dict) else {}, 'opensmile', song_feat)
            if isinstance(MODEL_OPENSMILE, dict):
                res = {"error": "OpenSMILE model is a state_dict; could not instantiate architecture automatically."}
            else:
                raw = predict_with_model(MODEL_OPENSMILE, eeg_pre, song_feat)
                res = normalize_prediction(raw)
        else:
            res = {"error": f"Song id '{song_id}' not found in OpenSMILE features CSV"}
        results["GESSNet + OpenSMILE"] = res
    else:
        results["GESSNet + OpenSMILE"] = {"error": "OpenSMILE model not loaded"}

    # YAMNet model prediction
    if MODEL_YAMNET is not None:
        if song_id in YAMNET_FEATURES:
            song_feat = np.array(list(YAMNET_FEATURES[song_id].values()), dtype=np.float32)
            MODEL_YAMNET = _ensure_model_is_callable(MODEL_YAMNET, MODEL_YAMNET if isinstance(MODEL_YAMNET, dict) else {}, 'yamnet', song_feat)
            if isinstance(MODEL_YAMNET, dict):
                res = {"error": "YAMNet model is a state_dict; could not instantiate architecture automatically."}
            else:
                raw = predict_with_model(MODEL_YAMNET, eeg_pre, song_feat)
                res = normalize_prediction(raw)
        else:
            res = {"error": f"Song id '{song_id}' not found in YAMNet features CSV"}
        results["GESSNet + YAMNet"] = res
    else:
        results["GESSNet + YAMNet"] = {"error": "YAMNet model not loaded"}

    # return JSON and also render page
    if request.headers.get("Accept", "") == "application/json" or request.is_json:
        return jsonify(results)
    return render_template("index.html", results=results, song_id=song_id)


@APP.route("/status", methods=["GET"])
def status():
    """Return diagnostic information about loaded models.

    For each model, report whether it's a callable or a state_dict. If it's a
    state_dict, include a sample of keys and shapes and attempt to instantiate
    the architecture while capturing any error message.
    """
    def inspect_model(obj, name):
        info = {"name": name}
        if obj is None:
            info["status"] = "not_loaded"
            return info
        if isinstance(obj, dict):
            info["status"] = "state_dict"
            # sample keys and shapes
            keys = list(obj.keys())[:40]
            info["keys_sample"] = keys
            info["shapes"] = {k: getattr(v, 'shape', None) for k, v in list(obj.items())[:40]}
            # try to instantiate via model_arch
            arch_path = os.path.join(BASE_DIR, "model_arch.py")
            if os.path.exists(arch_path):
                try:
                    import importlib.util, inspect
                    spec = importlib.util.spec_from_file_location("model_arch", arch_path)
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    # choose factory
                    base = name.lower()
                    factory = getattr(mod, 'get_model', None)
                    if factory is None:
                        if 'opensmile' in base and hasattr(mod, 'get_model_opensmile'):
                            factory = getattr(mod, 'get_model_opensmile')
                        elif ('yamnet' in base or 'yam' in base) and hasattr(mod, 'get_model_yamnet'):
                            factory = getattr(mod, 'get_model_yamnet')

                    song_feature_dim = None
                    num_classes = None
                    input_channels = None
                    try:
                        for k, v in obj.items():
                            if 'song_project.0.weight' in k or 'song_attention.0.weight' in k:
                                if hasattr(v, 'shape') and len(v.shape) >= 2:
                                    song_feature_dim = v.shape[1]
                            if 'fc_combined' in k and getattr(v, 'ndim', 0) == 2:
                                num_classes = v.shape[0]
                            if 'depthwiseConv.0.weight' in k and hasattr(v, 'shape'):
                                # weight shape is (out_ch, 1, in_ch, 1)
                                if len(v.shape) >= 3:
                                    input_channels = v.shape[2]
                    except Exception:
                        pass

                    if factory is None:
                        info['instantiate_attempt'] = 'no_factory_found'
                    else:
                        try:
                            # try factory with inferred shapes including input_channels
                            try:
                                m = factory(name, song_feature_dim=song_feature_dim, num_classes=num_classes, input_channels=input_channels)
                            except TypeError:
                                try:
                                    m = factory(song_feature_dim=song_feature_dim, num_classes=num_classes, input_channels=input_channels)
                                except Exception:
                                    m = factory()
                            try:
                                m.load_state_dict(obj)
                                info['instantiate_attempt'] = 'success'
                                info['model_type'] = str(type(m))
                            except Exception as e:
                                info['instantiate_attempt'] = 'load_state_failed'
                                info['instantiate_error'] = str(e)
                        except Exception as e:
                            info['instantiate_attempt'] = 'factory_call_failed'
                            info['instantiate_error'] = str(e)
                except Exception as e:
                    info['inspect_error'] = str(e)
            return info
        else:
            info["status"] = "callable"
            info["type"] = str(type(obj))
            try:
                import inspect
                info["callable"] = callable(obj)
                info["params"] = sum(p.numel() for p in obj.parameters()) if hasattr(obj, 'parameters') else None
            except Exception:
                pass
            return info

    out = {
        "opensmile": inspect_model(MODEL_OPENSMILE, 'opensmile'),
        "yamnet": inspect_model(MODEL_YAMNET, 'yamnet')
    }
    return jsonify(out)


if __name__ == "__main__":
    APP.run(host="0.0.0.0", port=5002, debug=True)
