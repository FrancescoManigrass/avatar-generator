#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import logging
import os
import sys
from pathlib import Path
import argparse
from typing import Any, Dict, Callable, Optional, List, Union

import numpy as np
import torch
from PIL import Image
from joblib import load

from measurement_evaluator import Human



# ----------------------------- Utils base -----------------------------


def setup_logging(level: str = "INFO"):
    print(f"[DEBUG] setup_logging: configuro logging con livello {level}")
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(p: Path) -> Path:
    print(f"[DEBUG] ensure_dir: controllo/creo directory {p}")
    p.mkdir(parents=True, exist_ok=True)
    return p


def expand_path(p: Union[str, Path]) -> Path:
    print(f"[DEBUG] expand_path: espando path {p}")
    return Path(os.path.expandvars(str(p))).expanduser()


def pil_to_resized_gray(path: Path, target_size: int = 512) -> np.ndarray:
    img = Image.open(path).convert("L").resize((target_size, target_size))
    return np.array(img, dtype="float32")


def normalize_silhouette_like(arr_gray_0_255: np.ndarray) -> np.ndarray:
    arr = arr_gray_0_255 / 255.0
    arr[arr < 1.0] = 0.0
    return 1.0 - arr


def to_float01(mask_uint8: np.ndarray) -> np.ndarray:
    m = mask_uint8.astype("float32")
    if m.max() > 1.0:
        m /= 255.0
    m = (m > 0.5).astype("float32")
    return m


# ----------------------------- Segmentazione (plugin + fallback) -----------------------------


def dynamic_import_func(script_path: Path, func_name: str) -> Callable:
    import importlib.util

    spec = importlib.util.spec_from_file_location("people_segmentation_plugin", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossibile caricare plugin da {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise AttributeError(f"Funzione '{func_name}' non trovata in {script_path}")
    return fn


def run_plugin_segmenter(
    image_path: Path,
    func: Callable,
    expects_path: Optional[bool] = None,
) -> np.ndarray:
    if expects_path is True:
        mask = func(str(image_path))
    else:
        try:
            import cv2

            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Impossibile leggere immagine: {image_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = func(rgb)
        except TypeError:
            mask = func(str(image_path))

    mask = np.array(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype(np.uint8) * 255
    return mask


def run_grabcut(image_path: Path, rect_shrink: float = 0.01, iter_count: int = 5) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Impossibile leggere immagine: {image_path}")
    h, w = bgr.shape[:2]
    dx, dy = int(w * rect_shrink), int(h * rect_shrink)
    rect = (dx, dy, w - 2 * dx, h - 2 * dy)
    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    import cv2 as _cv2

    _cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, iter_count, _cv2.GC_INIT_WITH_RECT)
    mask = np.where((mask == 1) | (mask == 3), 255, 0).astype("uint8")
    return mask


def load_model_with_human(path: Path):
    """
    Carica un modello joblib assicurandosi che 'Human' sia visibile come __main__.Human,
    per compatibilità con i pickle salvati in vecchi script.
    """
    print(f"[DEBUG] load_model_with_human: carico modello da {path}")
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "Human"):
        # Espone la classe Human anche come __main__.Human
        setattr(main_mod, "Human", Human)

    return load(path)

def load_deep2dencoder(calvis_dataset: str):

    import importlib.util

    calvis_dataset = str(calvis_dataset).strip().lower()
    print(f"[DEBUG] load_deep2dencoder: dataset calvis = {calvis_dataset}")

    # 🔹 nel caso SMPL
    if calvis_dataset == "smpl":
        print("carico smpl")
        model_path = Path(
            r"C:\Users\lab2O\Documents\Francesco Manigrasso\polito\Tesi moro\3DAvatarGenerator\humanet-master\utils\model.py"
        )
        label = "SMPL (3DAvatarGenerator\\utils\\model.py)"
    # 🔹 nel caso CALVIS (tuo dataset "calvis")
    else:
        print("carico supr")
        model_path = Path(

            r"C:\Users\lab2O\Documents\Francesco Manigrasso\polito\Tesi moro\3DAvatarGenerator\utils\model.py"
        )
        label = "CALVIS (humanet-master\\utils\\model.py)"

    if not model_path.is_file():
        raise FileNotFoundError(f"File model.py non trovato: {model_path}")

    logging.info(f"Carico Deep2DEncoder da {model_path} [{label}]")

    spec = importlib.util.spec_from_file_location("dyn_deep2d_model", model_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossibile caricare modulo da {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore

    Deep2DEncoder = getattr(module, "Deep2DEncoder", None)
    if Deep2DEncoder is None:
        raise AttributeError(f"Classe Deep2DEncoder non trovata in {model_path}")

    return Deep2DEncoder


def segment_image_to_mask_old(
    image_path: Path,
    target_size: int,
    seg_cfg: Dict[str, Any],
    debug_out_dir: Optional[Path],
    debug_prefix: str,
) -> np.ndarray:
    print(
        f"[DEBUG] segment_image_to_mask_old: immagine {image_path}, target_size={target_size}, debug_prefix={debug_prefix}, cfg={seg_cfg}"
    )
    mask_u8 = None
    if seg_cfg.get("enabled", False):
        try:
            script = expand_path(seg_cfg.get("script", "people_segmentation.py"))
            func_name = seg_cfg.get("function", "segment")
            expects_path = seg_cfg.get("expects_path")
            fn = dynamic_import_func(script, func_name)
            mask_u8 = run_plugin_segmenter(image_path, fn, expects_path=expects_path)
            logging.info(f"Segmentazione via plugin OK: {script}::{func_name}")
        except Exception as e:
            if not seg_cfg.get("allow_grabcut_fallback", True):
                raise
            logging.warning(f"Plugin segmentazione fallito ({e!r}). Uso GrabCut di fallback...")

    if mask_u8 is None:
        mask_u8 = run_grabcut(image_path)

    mask_img = Image.fromarray(mask_u8).resize((target_size, target_size), Image.NEAREST)
    mask_u8 = np.array(mask_img, dtype=np.uint8)
    mask = to_float01(mask_u8)

    if debug_out_dir is not None:
        try:
            import cv2

            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is not None:
                vis = cv2.addWeighted(
                    bgr,
                    1.0,
                    (cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2BGR) * (0, 255, 0)).astype(np.uint8),
                    0.5,
                    0.0,
                )
                cv2.imwrite(str(debug_out_dir / f"{debug_prefix}_overlay.png"), vis)
            Image.fromarray((mask * 255).astype(np.uint8)).save(
                debug_out_dir / f"{debug_prefix}_mask.png"
            )
        except Exception as e:
            logging.debug(f"Salvataggi debug non riusciti: {e!r}")

    return mask


# ----------------------------- Segmentazione (plugin + fallback, migliorata) -----------------------------


def dynamic_import_func(script_path: Path, func_name: str) -> Callable:
    import importlib.util

    spec = importlib.util.spec_from_file_location("people_segmentation_plugin", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossibile caricare plugin da {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    fn = getattr(module, func_name, None)
    if not callable(fn):
        raise AttributeError(f"Funzione '{func_name}' non trovata in {script_path}")
    return fn


def run_plugin_segmenter(
    image_path: Path,
    func: Callable,
    expects_path: Optional[bool] = None,
) -> np.ndarray:
    """
    Esegue il plugin di segmentazione:
    - se expects_path=True: gli passa una stringa (path)
    - altrimenti prova prima con immagine RGB (numpy), poi con path in fallback.
    Ritorna sempre una mask uint8 {0,255}.
    """
    if expects_path is True:
        mask = func(str(image_path))
    else:
        try:
            import cv2

            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"Impossibile leggere immagine: {image_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = func(rgb)
        except TypeError:
            # il plugin vuole il path
            mask = func(str(image_path))

    mask = np.array(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]

    # forza binario {0,255}
    if mask.dtype != np.uint8:
        mask = (mask > 0.5).astype("uint8") * 255

    return mask


def run_grabcut(image_path: Path, rect_shrink: float = 0.01, iter_count: int = 5) -> np.ndarray:
    """
    Segmentazione fallback con GrabCut, resa più robusta:
    - rettangolo centrale più stretto (meno sfondo)
    - tieni solo il componente connesso più grande (persona)
    - morfologia (close + open) per pulire bordi e buchi
    """
    import cv2

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Impossibile leggere immagine: {image_path}")

    h, w = bgr.shape[:2]

    # rettangolo "interno" per ridurre lo sfondo
    dx, dy = int(w * rect_shrink), int(h * rect_shrink)
    rect = (dx, dy, w - 2 * dx, h - 2 * dy)

    mask = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(bgr, mask, rect, bgdModel, fgdModel, iter_count, cv2.GC_INIT_WITH_RECT)

    # foreground = etichette 1 o 3
    mask_fg = np.where((mask == 1) | (mask == 3), 1, 0).astype("uint8")

    # 🔹 prendi solo il componente connesso più grande
    num_labels, labels = cv2.connectedComponents(mask_fg)
    if num_labels > 1:
        # bincount per ogni etichetta (0 = background)
        counts = np.bincount(labels.flatten())
        counts[0] = 0  # ignora sfondo
        largest_label = int(np.argmax(counts))
        mask_fg = np.where(labels == largest_label, 1, 0).astype("uint8")

    # 🔹 morfologia per chiudere buchi e togliere rumore
    kernel = np.ones((7, 7), np.uint8)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_fg = cv2.morphologyEx(mask_fg, cv2.MORPH_OPEN, kernel, iterations=1)

    return (mask_fg * 255).astype("uint8")


def segment_image_to_mask(
    image_path: Path,
    target_size: int,
    seg_cfg: Dict[str, Any],
    debug_out_dir: Optional[Path],
    debug_prefix: str,
) -> np.ndarray:
    """
    Restituisce una mask normalizzata in [0,1] (float32) di dimensione target_size x target_size.
    - se il plugin funziona → usa il plugin
    - altrimenti → usa GrabCut migliorato
    - post-processing morfologico dopo il resize
    - salva sia mask che overlay in debug_out_dir (se presente)
    """
    print(
        f"[DEBUG] segment_image_to_mask: image_path={image_path}, target_size={target_size}, debug_prefix={debug_prefix}, seg_cfg={seg_cfg}"
    )
    mask_u8: Optional[np.ndarray] = None

    # ----------------- 1) Tentativo con plugin -----------------
    if seg_cfg.get("enabled", False):
        try:
            script = expand_path(seg_cfg.get("script", "people_segmentation.py"))
            func_name = seg_cfg.get("function", "segment")
            expects_path = seg_cfg.get("expects_path")
            fn = dynamic_import_func(script, func_name)
            mask_u8 = run_plugin_segmenter(image_path, fn, expects_path=expects_path)
            logging.info(f"Segmentazione via plugin OK: {script}::{func_name}")
            print("[DEBUG] segment_image_to_mask: segmentazione plugin riuscita")
        except Exception as e:
            missing_pkg_hint = ""
            if isinstance(e, ModuleNotFoundError) and "people_segmentation.pre_trained_models" in str(e):
                missing_pkg_hint = " Assicurati di avere il pacchetto 'people-segmentation' installato."

            logging.warning(f"Plugin segmentazione fallito ({e!r}).{missing_pkg_hint}")
            print(
                f"[DEBUG] segment_image_to_mask: plugin fallito con errore {e!r}.{missing_pkg_hint}"
            )

            if not seg_cfg.get("allow_grabcut_fallback", True):
                # se non è permesso usare GrabCut, rilancia l'errore
                raise

    # ----------------- 2) Fallback con GrabCut -----------------
    if mask_u8 is None:
        logging.info("Uso GrabCut di fallback per la segmentazione...")
        print("[DEBUG] segment_image_to_mask: uso GrabCut di fallback")
        mask_u8 = run_grabcut(image_path)

    # ----------------- 3) Resize e pulizia morfologica -----------------
    # Resize a target_size x target_size
    mask_img = Image.fromarray(mask_u8).resize((target_size, target_size), Image.NEAREST)
    mask_u8 = np.array(mask_img, dtype=np.uint8)

    # Piccola morfologia extra dopo il resize
    try:
        import cv2

        kernel = np.ones((5, 5), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    except Exception as e:
        logging.debug(f"Morfologia su mask ridimensionata non riuscita ({debug_prefix}): {e!r}")
        print(f"[DEBUG] segment_image_to_mask: morfologia fallita con errore {e!r}")

    # Converti in [0,1] float32 (binaria)
    mask = to_float01(mask_u8)

    # ----------------- 4) Salvataggio debug (mask + overlay) -----------------
    if debug_out_dir is not None:
        debug_out_dir.mkdir(parents=True, exist_ok=True)

        # 4.1) Salva la mask pura (questa è quella che poi mostri in Gradio)
        try:
            mask_path = debug_out_dir / f"{debug_prefix}_mask.png"
            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            print(f"[DEBUG] segment_image_to_mask: mask salvata in {mask_path}")
        except Exception as e:
            logging.debug(f"Salvataggio mask debug non riuscito ({debug_prefix}): {e!r}")
            print(f"[DEBUG] segment_image_to_mask: errore nel salvataggio mask {e!r}")

        # 4.2) Prova a salvare anche l'overlay (solo per ispezione visiva)
        try:
            import cv2

            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is not None:
                color_mask = np.zeros_like(bgr)
                # canale verde = mask
                color_mask[..., 1] = mask_u8
                overlay = cv2.addWeighted(bgr, 0.7, color_mask, 0.3, 0.0)
                overlay_path = debug_out_dir / f"{debug_prefix}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
                print(f"[DEBUG] segment_image_to_mask: overlay salvata in {overlay_path}")
        except Exception as e:
            logging.debug(f"Salvataggio overlay debug non riuscito ({debug_prefix}): {e!r}")
            print(f"[DEBUG] segment_image_to_mask: errore nel salvataggio overlay {e!r}")

    return mask


# ----------------------------- Feature extractor -----------------------------




def extract_features_ae(
    front: np.ndarray,
    side: np.ndarray,
    ckpt_path: Path,
    calvis_dataset: str,
) -> np.ndarray:
    print(
        "[DEBUG] extract_features_ae: estraggo feature AE",
        {"ckpt_path": ckpt_path, "calvis_dataset": calvis_dataset, "front_shape": getattr(front, "shape", None), "side_shape": getattr(side, "shape", None)},
    )
    device = torch.device("cpu")

    # Prende la classe giusta in base a smpl / calvis
    Deep2DEncoder = load_deep2dencoder(calvis_dataset)

    encoder = Deep2DEncoder(image_size=512, kernel_size=3, n_filters=32)
    state = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(state)
    encoder.eval()
    encoder.requires_grad_(False)

    with torch.no_grad():
        f = torch.tensor(front, dtype=torch.float32).view(1, 1, 512, 512)
        s = torch.tensor(side, dtype=torch.float32).view(1, 1, 512, 512)
        f_feat = encoder(f).detach().cpu().numpy().reshape(-1)
        s_feat = encoder(s).detach().cpu().numpy().reshape(-1)

    return np.concatenate([f_feat, s_feat], axis=-1).reshape(1, -1)

def extract_features_pca(front: np.ndarray, side: np.ndarray, pca_dir: Path, gender: str) -> np.ndarray:
    print(
        "[DEBUG] extract_features_pca: estraggo feature PCA",
        {"pca_dir": pca_dir, "gender": gender, "front_shape": getattr(front, "shape", None), "side_shape": getattr(side, "shape", None)},
    )
    pca_front_path = pca_dir / f"pca_{gender}_front.joblib"
    pca_side_path = pca_dir / f"pca_{gender}_side.joblib"
    if not pca_front_path.is_file() or not pca_side_path.is_file():
        raise FileNotFoundError("File PCA non trovati. Controlla 'pca_dir' e 'gender' nel config.")
    pca_front = load(pca_front_path)
    pca_side = load(pca_side_path)
    f_feat = pca_front.transform(front.reshape(1, 512 * 512))
    s_feat = pca_side.transform(side.reshape(1, 512 * 512))
    return np.concatenate([np.array(f_feat), np.array(s_feat)], axis=-1)


def append_body_metrics(
    features: np.ndarray, height: float, weight: Optional[float], measurement_model: str
) -> np.ndarray:
    print(
        "[DEBUG] append_body_metrics: aggiungo metriche corporee",
        {"features_shape": getattr(features, "shape", None), "height": height, "weight": weight, "measurement_model": measurement_model},
    )
    h = np.array(height, dtype="float32").reshape(1, 1)
    if measurement_model == "nomo":
        return np.concatenate([features, h], axis=-1)
    w = np.array(weight, dtype="float32").reshape(1, 1)
    return np.concatenate([features, h, w], axis=-1)


# ----------------------------- Path helper -----------------------------


def resolve_paths(
    data_root: Path,
    parameters: str,
    split: str,
    gender: str,
    lr: str,
    ae_ckpt_name: str,
    calvis_dataset: str = "supr",
) -> Dict[str, Path]:
    """
    - calvis_dataset = 'supr' -> layout originale data{parameters}
    - calvis_dataset = 'smpl' -> usa solo humanet-master/weights/feature_extractor_{gender}_50.pth
    """
    # Normalizza le stringhe
    calvis_dataset = str(calvis_dataset).strip().lower()
    split = str(split).strip().lower()

    # 🔹 CASO SMPL → IGNORA parameters/split/lr e punta SEMPRE a .../weights/...
    if calvis_dataset == "smpl" or "smpl" in split:
        weights_dir = data_root / "weights"
        ae_ckpt = weights_dir / ae_ckpt_name
        logging.info(f"[resolve_paths] SMPL: uso checkpoint AE in: {ae_ckpt}")
        return {
            "ae_ckpt": ae_ckpt,
        }

    # 🔹 CASO STANDARD "SUPR"
    base = data_root / f"data{parameters}"
    return {
        "ae_ckpt": base / split / "weights2" / lr / ae_ckpt_name,
        "x_train_features": base / split / "features2" / gender / "ae_train_features.npy",
        "x_train_hw": base
        / split
        / "dataloaders"
        / gender
        / f"train_h_w_measures_{gender}_density.npy",
        "y_shape_train": base / split / "dataloaders" / gender / "train_betas.npy",
        "template": base / f"{gender}_template.npy",
        "shape_dirs": base / f"{gender}_shapedirs.npy",
        "faces": base / "faces.npy",
    }

def must_exist(label: str, path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"{label} non trovato: {path}")


# ----------------------------- Config -----------------------------


REQUIRED_KEYS = [
    "experiment",
    "front_img",
    "side_img",
    "gender",
    "height",
    "weight",
    "feature_model",
    "measurement_model",
    "mesh_name",
    "output_dir",
    "data_root",
    "split",
    "lr",
    "ae_ckpt_name",
    "pca_dir",
    "nomo_dir",
    "segmentation",
]

PATH_KEYS = ["front_img", "side_img", "output_dir", "data_root", "pca_dir", "nomo_dir"]

_DEFAULT_PARAMS = [10, 16, 32, 64, 128, 256, 300]




def load_config_file(cfg_path: Path) -> Dict[str, Any]:
    if cfg_path.suffix.lower() in {".json"}:
        return json.loads(Path(cfg_path).read_text(encoding="utf-8"))
    elif cfg_path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("Per usare YAML installa 'pyyaml' oppure usa un config JSON.") from e
        return yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    else:
        raise ValueError("Formato config non supportato. Usa .json, .yml o .yaml")


def normalize_and_validate(cfg: Dict[str, Any]) -> Dict[str, Any]:
    print("[DEBUG] normalize_and_validate: config in ingresso", cfg)
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"Chiavi mancanti nel config: {missing}")

    for k in PATH_KEYS:
        cfg[k] = expand_path(cfg[k])

    cfg["height"] = float(cfg["height"])
    cfg["weight"] = float(cfg["weight"])
    cfg["experiment"] = str(cfg["experiment"])
    cfg["gender"] = str(cfg["gender"])
    cfg["feature_model"] = str(cfg["feature_model"])
    cfg["measurement_model"] = str(cfg["measurement_model"])
    cfg["mesh_name"] = str(cfg["mesh_name"])
    cfg["split"] = str(cfg["split"])
    cfg["lr"] = str(cfg["lr"])
    cfg["ae_ckpt_name"] = str(cfg["ae_ckpt_name"])
    cfg["log_level"] = str(cfg.get("log_level", "INFO"))

    # tipo dataset CALVIS: supr (default) o smpl
    cfg["calvis_dataset"] = str(cfg.get("calvis_dataset", "supr")).strip().lower()


    seg = cfg.get("segmentation", {}) or {}
    seg.setdefault("enabled", False)
    seg.setdefault("script", "people_segmentation.py")
    seg.setdefault("function", "segment")
    seg.setdefault("expects_path", None)
    seg.setdefault("allow_grabcut_fallback", True)
    seg.setdefault("save_debug", True)
    cfg["segmentation"] = seg

    params_list = cfg.get("parameters_list", _DEFAULT_PARAMS)
    if not isinstance(params_list, (list, tuple)) or not all(int(p) == p for p in params_list):
        raise ValueError("parameters_list deve essere una lista di interi (es. [10,16,32,...]).")
    cfg["parameters_list"] = [int(p) for p in params_list]

    print("[DEBUG] normalize_and_validate: config normalizzata", cfg)

    return cfg


# ----------------------------- Pipeline per un singolo 'parameters' -----------------------------


def run_for_parameters(cfg: Dict[str, Any], parameters_value: int, target_size: int = 512) -> Path:
    """
    Esegue l'intera pipeline per uno specifico 'parameters'.
    """
    print(
        f"[DEBUG] run_for_parameters: start con parameters_value={parameters_value}, target_size={target_size}"
    )
    print("[DEBUG] run_for_parameters: config ricevuta", cfg)
    calvis_dataset = cfg.get("calvis_dataset", "supr")
    # Output: .../experiment/p{parameters}/
    exp_base = ensure_dir(Path(cfg["output_dir"]) / cfg["experiment"])
    exp_out_dir = ensure_dir(exp_base / f"p{parameters_value}")
    dbg_dir = exp_out_dir / "debug"
    if cfg["segmentation"].get("save_debug", True):
        ensure_dir(dbg_dir)

    # --- Segmentazione / Preproc ---
    logging.info(f"[p{parameters_value}] Segmento e preprocesso le immagini...")
    print(
        f"[DEBUG] run_for_parameters: segmentazione front={cfg['front_img']} side={cfg['side_img']} enabled={cfg['segmentation']['enabled']}"
    )
    front_mask = (
        segment_image_to_mask(
            image_path=Path(cfg["front_img"]),
            target_size=target_size,
            seg_cfg=cfg["segmentation"],
            debug_out_dir=(dbg_dir if cfg["segmentation"].get("save_debug", True) else None),
            debug_prefix="front",
        )
        if cfg["segmentation"]["enabled"]
        else None
    )

    side_mask = (
        segment_image_to_mask(
            image_path=Path(cfg["side_img"]),
            target_size=target_size,
            seg_cfg=cfg["segmentation"],
            debug_out_dir=(dbg_dir if cfg["segmentation"].get("save_debug", True) else None),
            debug_prefix="side",
        )
        if cfg["segmentation"]["enabled"]
        else None
    )

    front = (
        front_mask
        if front_mask is not None
        else normalize_silhouette_like(pil_to_resized_gray(Path(cfg["front_img"]), target_size))
    )
    side = (
        side_mask
        if side_mask is not None
        else normalize_silhouette_like(pil_to_resized_gray(Path(cfg["side_img"]), target_size))
    )

    print(
        "[DEBUG] run_for_parameters: silhouette preparate",
        {
            "front_mask_present": front_mask is not None,
            "side_mask_present": side_mask is not None,
            "front_shape": getattr(front, "shape", None),
            "side_shape": getattr(side, "shape", None),
        },
    )

    # --- Path per questo 'parameters' ---
    ae_ckpt_name = cfg["ae_ckpt_name"].format(gender=cfg["gender"])
    paths = resolve_paths(
        data_root=Path(cfg["data_root"]),
        parameters=str(parameters_value),
        split=cfg["split"],
        gender=cfg["gender"],
        lr=cfg["lr"],
        ae_ckpt_name=ae_ckpt_name,
        calvis_dataset=calvis_dataset,
    )
    print(f"[DEBUG] run_for_parameters: percorsi risolti {paths}")

    # --- Estrazione feature ---
    logging.info(f"[p{parameters_value}] Estrazione feature...")
    if cfg["feature_model"] == "ae":
        must_exist("Checkpoint AE", paths["ae_ckpt"])
        features = extract_features_ae(
            front,
            side,
            paths["ae_ckpt"],
            cfg.get("calvis_dataset", "calvis"),  # o "supr" se preferisci, l'importante è che per SMPL sia "smpl"
        )

    elif cfg["feature_model"] == "pca":
        features = extract_features_pca(front, side, Path(cfg["pca_dir"]), cfg["gender"])
    else:
        raise ValueError("feature_model non valido (usa 'ae' o 'pca').")

    features = append_body_metrics(features, cfg["height"], cfg["weight"], cfg["measurement_model"])
    print(
        "[DEBUG] run_for_parameters: feature finali pronte",
        {"shape": getattr(features, "shape", None), "measurement_model": cfg["measurement_model"]},
    )

    # --- Modello/fit/predizione ---
    logging.info(f"[p{parameters_value}] Costruisco/fitto il modello 'Human'...")

    if cfg["measurement_model"] == "nomo":
        nomo_path = Path(cfg["nomo_dir"]) / f"nomo_{cfg['gender']}_krr.pkl"
        must_exist("Modello NOMO", nomo_path)
        human = load_model_with_human(nomo_path)

    elif cfg["measurement_model"] == "calvis":
        if calvis_dataset == "smpl":
            model_path = Path(cfg["nomo_dir"]) / f"calvis_{cfg['gender']}_krr.pkl"
            must_exist("Modello CALVIS (SMPL)", model_path)
            human = load_model_with_human(model_path)
        else:
            # caso standard "supr": fitta il modello da dati
            for label in ["x_train_features", "x_train_hw", "y_shape_train", "template", "shape_dirs", "faces"]:
                must_exist(label, paths[label])

            X_train = np.load(paths["x_train_features"]).squeeze()
            X_train_h_w = np.load(paths["x_train_hw"], allow_pickle=True)
            X_train = np.concatenate([X_train, X_train_h_w], axis=-1)
            X_train = np.nan_to_num(X_train, copy=False)
            y_shape_train = np.load(paths["y_shape_train"])
            template = np.load(paths["template"])
            shape_dirs = np.load(paths["shape_dirs"])
            faces = np.load(paths["faces"])

            human = Human(
                kernel="polynomial",
                alpha=1,
                degree=3,
                template=template,
                shape_dirs=shape_dirs,
                faces=faces,
            )
            human.fit_shape(X_train, y_shape_train)
        print("[DEBUG] run_for_parameters: modello Human pronto", human)
    else:
        raise ValueError("measurement_model non valido (usa 'calvis' o 'nomo').")

    # --- Predizione shape + export mesh ---
    logging.info(f"[p{parameters_value}] Predico shape e genero mesh 3D...")
    shape = human.predict_shape(features)
    mesh = human.display_3D(shape)
    mesh_path = exp_out_dir / cfg["mesh_name"]
    mesh.export(mesh_path.as_posix())
    print("[DEBUG] run_for_parameters: mesh esportata", mesh_path)
    logging.info(f"[p{parameters_value}] 3D model salvato in: {mesh_path}")
    return mesh_path


# ----------------------------- Main (solo per uso da CLI) -----------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Avatar generation batch via config (multi-parameters)."
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path al file di configurazione (.json/.yaml)."
    )
    args = parser.parse_args()

    cfg_raw = load_config_file(args.config)
    cfg = normalize_and_validate(cfg_raw)
    setup_logging(cfg["log_level"])

    params: List[int] = cfg["parameters_list"]
    logging.info(f"Eseguo batch per parameters: {params}")

    exported: List[Any] = []
    for p in params:
        try:
            out_path = run_for_parameters(cfg, p)
            exported.append((p, out_path))
        except Exception as e:
            logging.error(f"[p{p}] ERRORE: {e}", exc_info=True)

    print("\n=== RISULTATI ESPORTAZIONE ===")
    for p, path in exported:
        print(f"p={p}: {path}")
    print("===============================")


if __name__ == "__main__":
    main()
