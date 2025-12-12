# Avatar Generator

Avatar Generator is a toolkit for generating 3D OBJ models from front and side images of the human body. It includes a Gradio-based web UI and a configurable CLI pipeline that combine 2D feature extraction, body measurement regression, and mesh reconstruction through the CALVIS and NOMO models.

## Key Features
- **Gradio web interface** (`app.py`): upload front/side images, choose the model (CALVIS/NOMO), enable or disable automatic segmentation, and download the generated OBJ with a 3D preview and segmentation mask overlays.
- **Modular pipeline** (`avatar_from_config.py`): normalizes and validates a configuration file, optionally applies segmentation plugins or a GrabCut fallback, extracts features (autoencoder or PCA), estimates shape with the selected regression model, and saves the mesh.
- **Multiple dataset support**: predefined presets for CALVIS (original “supr” dataset or SMPL variant with 10 parameters) plus configurable paths for weights, PCA, NOMO assets, and training checkpoints.

## Requirements
- Python 3.8+ (tested with PyTorch and OpenCV).
- Base dependencies: `pip install -r requirements.txt`. Add `gradio` and, if you use YAML configs, `pyyaml`.
- Paths to weights and datasets must be customized: the defaults in `app.py` and `avatar_from_config.py` point to sample local directories and should be updated to your assets (AE checkpoints, PCA weights, NOMO models, CALVIS/HUMANET datasets).

## Quick Start (Gradio UI)
1. Update the default paths in `DEFAULT_CONFIG` or `CALVIS_SMPL_PRESET` inside `app.py` to point to your data and weights.
2. Install dependencies.
3. Start the interface:
   ```bash
   python app.py
   ```
4. Upload 512x512 front and side images, choose `measurement_model` (CALVIS or NOMO) and the CALVIS dataset (`supr`/`smpl`). If `measurement_model=calvis` and `dataset=smpl`, the parameters are forced to 10.
5. Download the generated OBJ or view it in the built-in viewer; if segmentation is enabled, overlays are shown and masks are saved in `debug/`.

> Tip: set `GRADIO_SERVER_PORT` to choose the server port; if the port is busy the app automatically falls back to another.

## Command-Line Execution
Run the pipeline in batch mode with a JSON or YAML configuration file:
```bash
python avatar_from_config.py --config config.json
```
Minimal `config.json` example:
```json
{
  "experiment": "demo_cli",
  "front_img": "path/to/front.png",
  "side_img": "path/to/side.png",
  "gender": "female",
  "height": 1.70,
  "weight": 60.0,
  "feature_model": "ae",
  "measurement_model": "calvis",
  "mesh_name": "subject.obj",
  "parameters_list": [32],
  "output_dir": "./outputs",
  "data_root": "./dataset_root",
  "split": "train_test_data_fold1",
  "lr": "0.0001",
  "ae_ckpt_name": "new_base_feature_extractor_{gender}_200.pth",
  "pca_dir": "./weights",
  "nomo_dir": "./weights",
  "segmentation": {
    "enabled": true,
    "script": "people_segmentation.py",
    "function": "segment",
    "expects_path": null,
    "allow_grabcut_fallback": true,
    "save_debug": true
  }
}
```
The command runs the pipeline for every value in `parameters_list`, saves meshes in `output_dir/experiment/p{parameters}/`, and prints a summary of the generated paths.

## Segmentation and Fallback Notes
- If segmentation is enabled, the code dynamically imports the specified plugin (`script` and `function`) and uses the returned masks; if an issue occurs or `allow_grabcut_fallback` is `true`, OpenCV GrabCut is applied.
- Masks and overlays are saved when `save_debug` is `true`, which is useful for debugging the pipeline.

## Project Structure
- `app.py`: Gradio interface and configuration assembly from user inputs.
- `avatar_from_config.py`: pipeline implementation, configuration normalization, and CLI batch entry point.
- `utils/`, `humanet-master/`, `weights/` (if present): models, checkpoints, and helper scripts for feature extraction and regression.

## License
This project is distributed under the license specified in the `LICENSE` file.
