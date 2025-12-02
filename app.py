import os
from pathlib import Path

import gradio as gr

from avatar_from_config import (
    run_for_parameters,
    normalize_and_validate,
    setup_logging,
)

# --------------------------------------------------------
# Config base (dataset CALVIS "supr" = tuo dataset originale)
# --------------------------------------------------------
DEFAULT_CONFIG = {
    "experiment": "demo_webui",
    "front_img": "",
    "side_img": "",
    "gender": "female",
    "height": 1.70,
    "weight": 60.0,
    "feature_model": "ae",
    "measurement_model": "calvis",
    "mesh_name": "subject.obj",
    "parameters_list": [10],

    # 🔧 ADATTA QUESTI SE SERVE (dataset "supr")
    "output_dir": r"calvis_data\demo_supr",
    "data_root": r"D:\Francesco Manigrasso\avatargenerator\Dataset",
    "split": "train_test_data_fold1",
    "lr": "0.0001",
    "ae_ckpt_name": "new_base_feature_extractor_{gender}_200.pth",

    # directory comuni (per PCA, NOMO)
    "pca_dir": r"weights",
    "nomo_dir": r"weights",

    "segmentation": {
        "enabled": True,
        # Usa di default lo script originale basato su "people_segmentation" (richiede il pacchetto dedicato)
        "script": "people_segmentation.py",
        "function": "segment",
        "expects_path": None,
        "allow_grabcut_fallback": True,
        "save_debug": True,
    },
    "log_level": "INFO",
    "calvis_dataset": "smpl",
}

# --------------------------------------------------------
# Override per dataset CALVIS basato su SMPL (10 parametri)
# pesi:
#  calvis_female_krr.pkl / calvis_male_krr.pkl
#  feature_extractor_female_50.pth / feature_extractor_male_50.pth
# --------------------------------------------------------
CALVIS_SMPL_PRESET = {
    "data_root": r"C:\Users\lab2O\Documents\Francesco Manigrasso\polito\Tesi moro\3DAvatarGenerator\humanet-master",
    "split": "smpl_10params",    # non usato da smpl, ma lo lasciamo
    "lr": "0.0001",
    "ae_ckpt_name": "feature_extractor_{gender}_50.pth",
    "pca_dir": r"C:\Users\lab2O\Documents\Francesco Manigrasso\polito\Tesi moro\3DAvatarGenerator\humanet-master\weights",
    "nomo_dir": r"C:\Users\lab2O\Documents\Francesco Manigrasso\polito\Tesi moro\3DAvatarGenerator\humanet-master\weights",
    "output_dir": r"calvis_data\demo_smpl",
    "calvis_dataset": "smpl",
}

PARAMETERS_OPTIONS = [10, 16, 32, 64, 128, 256, 300]
FEATURE_MODELS = ["ae", "pca"]
MEASUREMENT_MODELS = ["calvis", "nomo"]

CALVIS_DATASET_CHOICES = ["supr", "smpl"]


def build_cfg(
    front_path: str,
    side_path: str,
    gender: str,
    height: float,
    weight: float,
    feature_model: str,
    measurement_model: str,
    parameters: int,
    experiment_name: str,
    calvis_dataset: str,
    segmentation_enabled: bool,
):
    print("[DEBUG] build_cfg chiamata", {
        "front_path": front_path,
        "side_path": side_path,
        "gender": gender,
        "height": height,
        "weight": weight,
        "feature_model": feature_model,
        "measurement_model": measurement_model,
        "parameters": parameters,
        "experiment_name": experiment_name,
        "calvis_dataset": calvis_dataset,
        "segmentation_enabled": segmentation_enabled,
    })
    cfg = DEFAULT_CONFIG.copy()

    # normalizza dataset
    calvis_dataset = (calvis_dataset or "supr").strip().lower()

    # se CALVIS + SMPL → override con humanet-master
    if measurement_model == "calvis" and calvis_dataset == "smpl":
        print("[DEBUG] build_cfg: applico preset CALVIS_SMPL_PRESET per dataset smpl")
        cfg.update(CALVIS_SMPL_PRESET)
    else:
        cfg["calvis_dataset"] = calvis_dataset

    experiment_name = (experiment_name or "").strip() or cfg["experiment"]

    cfg.update(
        {
            "experiment": experiment_name,
            "front_img": front_path,
            "side_img": side_path,
            "gender": gender,
            "height": float(height),
            "weight": float(weight),
            "feature_model": feature_model,
            "measurement_model": measurement_model,
            "mesh_name": f"{experiment_name}_{gender}_{measurement_model}_{calvis_dataset}_p{parameters}.obj",
            "parameters_list": [int(parameters)],
            "output_dir": os.path.join(cfg["output_dir"], gender),
        }
    )

    cfg = normalize_and_validate(cfg)

    # ✅ collega il checkbox alla config
    cfg["segmentation"]["enabled"] = bool(segmentation_enabled)

    print("[DEBUG] build_cfg: configurazione finale normalizzata", cfg)

    return cfg


def generate_avatar(
    front_file,
    side_file,
    gender,
    height,
    weight,
    feature_model,
    measurement_model,
    parameters,
    experiment_name,
    calvis_dataset,
    segmentation_enabled,
):
    print("[DEBUG] generate_avatar avviata")
    if front_file is None or side_file is None:
        # >>> ora ritorniamo 5 valori (mesh, model3d, log, seg_front, seg_side)
        return None, None, "❗ Carica sia la foto frontale che quella di profilo.", None, None

    front_path = str(front_file)
    side_path = str(side_file)

    forced_parameters = int(parameters)
    note = ""

    # CALVIS + SMPL → sempre 10 parametri
    if measurement_model == "calvis" and calvis_dataset == "smpl":
        forced_parameters = 10
        note = " (forzato a 10 per dataset SMPL)"
        print(
            "[DEBUG] generate_avatar: measurement_model=calvis e dataset=smpl, forzo parameters a 10"
        )

    print(
        "[DEBUG] generate_avatar: preparo configurazione",
        {
            "front_path": front_path,
            "side_path": side_path,
            "gender": gender,
            "height": height,
            "weight": weight,
            "feature_model": feature_model,
            "measurement_model": measurement_model,
            "requested_parameters": parameters,
            "forced_parameters": forced_parameters,
            "experiment_name": experiment_name,
            "calvis_dataset": calvis_dataset,
            "segmentation_enabled": segmentation_enabled,
        },
    )

    try:
        cfg = build_cfg(
            front_path=front_path,
            side_path=side_path,
            gender=gender,
            height=height,
            weight=weight,
            feature_model=feature_model,
            measurement_model=measurement_model,
            parameters=forced_parameters,
            experiment_name=experiment_name,
            calvis_dataset=calvis_dataset,
            segmentation_enabled=segmentation_enabled,
        )

        setup_logging(cfg["log_level"])

        print("[DEBUG] generate_avatar: config pronta, lancio run_for_parameters")

        mesh_path = run_for_parameters(cfg, forced_parameters)
        mesh_path = Path(mesh_path)

        # >>> NUOVO: recupero le immagini di segmentazione salvate in debug/
        # >>> NUOVO: recupero le MASCHERE di segmentazione salvate in debug/
        seg_front_path = None
        seg_side_path = None
        if cfg["segmentation"]["enabled"] and cfg["segmentation"].get("save_debug", True):
            dbg_dir = Path(cfg["output_dir"]) / cfg["experiment"] / f"p{forced_parameters}" / "debug"

            front_mask = dbg_dir / "front_mask.png"
            side_mask = dbg_dir / "side_mask.png"

            if front_mask.is_file():
                seg_front_path = str(front_mask)
            if side_mask.is_file():
                seg_side_path = str(side_mask)

        print(
            "[DEBUG] generate_avatar: generazione completata",
            {
                "mesh_path": str(mesh_path),
                "seg_front_path": seg_front_path,
                "seg_side_path": seg_side_path,
                "message_note": note,
            },
        )

        msg = (
            f"✅ Avatar generato con successo.\n"
            f"- measurement_model = {measurement_model}\n"
            f"- dataset CALVIS = {calvis_dataset}\n"
            f"- parameters = {forced_parameters}{note}\n"
            f"- gender = {gender}\n"
            f"- file OBJ: {mesh_path}"
        )

        # >>> restituisco anche i path delle segmentazioni
        return str(mesh_path), str(mesh_path), msg, seg_front_path, seg_side_path

    except Exception as e:
        print("[DEBUG] generate_avatar: eccezione rilevata", repr(e))
        # >>> anche in errore ritorniamo 5 valori
        return None, None, f"❌ Errore durante la generazione: {repr(e)}", None, None


# --------------------------------------------------------
# UI Gradio
# --------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown(
        """
    # Avatar Generator UI
    Carica le immagini, scegli il modello (CALVIS/NOMO), il dataset (supr/smpl)
    e genera il tuo avatar 3D (OBJ) con anteprima **e visualizzazione della segmentazione**.
    """
    )

    with gr.Row():
        front_input = gr.Image(
            label="Immagine FRONT (512x512)",
            type="filepath",
        )
        side_input = gr.Image(
            label="Immagine SIDE (512x512)",
            type="filepath",
        )

    with gr.Row():
        gender_input = gr.Radio(
            choices=["female", "male"],
            value="female",
            label="Gender",
        )
        parameters_input = gr.Dropdown(
            choices=PARAMETERS_OPTIONS,
            value=32,
            label="parameters (data{p})",
        )
        feature_model_input = gr.Radio(
            choices=FEATURE_MODELS,
            value="ae",
            label="Feature model",
        )
        measurement_model_input = gr.Radio(
            choices=MEASUREMENT_MODELS,
            value="calvis",
            label="Measurement model",
        )

    with gr.Row():
        calvis_dataset_input = gr.Radio(
            choices=CALVIS_DATASET_CHOICES,
            value="supr",
            label="Dataset CALVIS",
            info=(
                "Usato solo quando measurement_model = 'calvis'.\n"
                " - supr: dataset originale\n"
                " - smpl: modello con 10 parametri (pesi in humanet-master/weights)"
            ),
        )

    with gr.Row():
        height_input = gr.Slider(
            minimum=1.40,
            maximum=2.10,
            step=0.01,
            value=1.70,
            label="Altezza (m)",
        )
        weight_input = gr.Slider(
            minimum=40,
            maximum=120,
            step=0.5,
            value=60.0,
            label="Peso (kg)",
        )
        experiment_input = gr.Textbox(
            value="prova1",
            label="Nome esperimento (cartella sotto output_dir)",
        )

    # ✅ toggle per segmentazione automatica
    segmentation_enabled_input = gr.Checkbox(
        label="Segmentazione automatica (persona dallo sfondo)",
        value=True,
        info="Se attivo, il codice prova a segmentare automaticamente la persona (plugin + GrabCut di fallback).",
    )

    generate_btn = gr.Button("Genera avatar 3D")

    mesh_output = gr.File(label="Scarica file OBJ")
    mesh_viewer = gr.Model3D(label="Anteprima 3D (OBJ)")
    log_output = gr.Textbox(
        label="Messaggi / Log",
        lines=6,
    )

    # >>> NUOVO: output per visualizzare i risultati di segmentazione
    seg_front_output = gr.Image(
        label="Segmentazione FRONT (overlay)",
        type="filepath",
    )
    seg_side_output = gr.Image(
        label="Segmentazione SIDE (overlay)",
        type="filepath",
    )

    generate_btn.click(
        fn=generate_avatar,
        inputs=[
            front_input,
            side_input,
            gender_input,
            height_input,
            weight_input,
            feature_model_input,
            measurement_model_input,
            parameters_input,
            experiment_input,
            calvis_dataset_input,
            segmentation_enabled_input,
        ],
        # >>> ora abbiamo 5 output
        outputs=[mesh_output, mesh_viewer, log_output, seg_front_output, seg_side_output],
    )

    # ✅ Funzione per mostrare/nascondere il dropdown dei parameters
    def toggle_parameters_visibility(calvis_dataset_choice):
        if calvis_dataset_choice == "smpl":
            # nasconde la dropdown quando il dataset è smpl
            return gr.update(visible=False)
        else:
            # la mostra per supr
            return gr.update(visible=True)

    calvis_dataset_input.change(
        fn=toggle_parameters_visibility,
        inputs=calvis_dataset_input,
        outputs=parameters_input,
    )


def resolve_server_port():
    env_port = os.getenv("GRADIO_SERVER_PORT")
    if env_port:
        try:
            port_value = int(env_port)
            print(f"[DEBUG] GRADIO_SERVER_PORT set, using port {port_value}")
            return port_value
        except ValueError:
            print(
                f"[DEBUG] Invalid GRADIO_SERVER_PORT='{env_port}', falling back to automatic selection"
            )
            return None

    print("[DEBUG] GRADIO_SERVER_PORT not set, defaulting to port 4343 (with fallback)")
    return 4343


if __name__ == "__main__":
    preferred_port = resolve_server_port()
    try:
        demo.launch(
            server_name="0.0.0.0",  # ascolta su tutte le interfacce
            server_port=preferred_port,  # usa la porta preferita o None per scelta automatica
            share=True,  # per ottenere un link pubblico
        )
    except OSError as error:
        if preferred_port is not None:
            print(
                f"[DEBUG] Port {preferred_port} unavailable ({error}); retrying with automatic port selection"
            )
            demo.launch(
                server_name="0.0.0.0",
                server_port=None,  # lascia scegliere a Gradio una porta libera
                share=True,
            )
        else:
            raise
