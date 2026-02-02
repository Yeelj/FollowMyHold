#!/usr/bin/env python3
"""
Lightweight demo launcher for FollowMyHold.

Usage:
  python app.py            # Gradio UI
  python app.py --cli ...  # CLI mode
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from glob import glob
from typing import Dict, Optional
import threading
import time


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


def _parse_env_file(path: str) -> Dict[str, str]:
    data: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            data[key.strip()] = val.strip().strip("\"").strip("'")
    return data


def _write_env_file(path: str, data: Dict[str, str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for key, value in data.items():
            f.write(f'{key}="{value}"\n')


def _infer_conda_sh() -> Optional[str]:
    conda_sh = os.environ.get("CONDA_SH")
    if conda_sh:
        return conda_sh
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe:
        base = os.path.abspath(os.path.join(os.path.dirname(conda_exe), ".."))
        candidate = os.path.join(base, "etc", "profile.d", "conda.sh")
        if os.path.isfile(candidate):
            return candidate
    return None


def _build_config(
    base_config: str,
    *,
    project_root: str,
    base_dir: str,
    image_path: Optional[str],
    split_path: Optional[str],
    conda_sh: Optional[str],
    env_name: Optional[str],
    run_inpaint: Optional[bool],
    suppress_warnings: Optional[bool],
) -> str:
    env = _parse_env_file(base_config)
    env["PROJECT_ROOT"] = project_root
    env["BASE_DIR"] = base_dir
    if image_path:
        env["IMAGE_PATH"] = image_path
        env.pop("SPLIT_PATH", None)
    if split_path:
        env["SPLIT_PATH"] = split_path
        env.pop("IMAGE_PATH", None)
    if conda_sh:
        env["CONDA_SH"] = conda_sh
    if env_name:
        env["ENV_NAME"] = env_name
    if run_inpaint is not None:
        env["RUN_INPAINT"] = "1" if run_inpaint else "0"
    if suppress_warnings is not None:
        env["FOHO_SUPPRESS_WARNINGS"] = "1" if suppress_warnings else "0"

    fd, tmp_path = tempfile.mkstemp(prefix="foho_", suffix=".env")
    os.close(fd)
    _write_env_file(tmp_path, env)
    return tmp_path


def _run_pipeline(config_path: str) -> None:
    from foho.configs import load_config
    from foho.main import run_pipeline

    cfg = load_config(config_path)
    run_pipeline(cfg)


def _find_first(pattern: str) -> Optional[str]:
    matches = sorted(glob(pattern))
    return matches[0] if matches else None


def _img_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    return base.split("_")[0].split(".")[0]


def run_cli(args: argparse.Namespace) -> None:
    base_config = args.config or os.path.join(REPO_ROOT, "configs", "pipeline.env")
    if not os.path.isfile(base_config):
        base_config = os.path.join(REPO_ROOT, "configs", "pipeline.env.example")

    conda_sh = _infer_conda_sh()
    if not conda_sh:
        raise RuntimeError("CONDA_SH not found. Please set CONDA_SH in your environment.")

    base_dir = args.base_dir or tempfile.mkdtemp(prefix="foho_outputs_")
    config_path = _build_config(
        base_config,
        project_root=REPO_ROOT,
        base_dir=base_dir,
        image_path=args.image,
        split_path=args.split_path,
        conda_sh=conda_sh,
        env_name=os.environ.get("ENV_NAME", "foho"),
        run_inpaint=args.run_inpaint,
        suppress_warnings=args.suppress_warnings,
    )
    try:
        _run_pipeline(config_path)
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)


def build_gradio(config_path: Optional[str] = None) -> "gr.Blocks":
    import gradio as gr

    def _submit(image, run_inpaint, suppress_warnings):
        conda_sh = _infer_conda_sh()
        if not conda_sh:
            yield ("CONDA_SH not found. Please set CONDA_SH in your environment.", None, None, None, None, None, None, None, None, None, None)
            return
        if config_path:
            base_config = config_path
        else:
            base_config = os.path.join(REPO_ROOT, "configs", "pipeline.env")
            if not os.path.isfile(base_config):
                base_config = os.path.join(REPO_ROOT, "configs", "pipeline.env.example")
        if not image:
            yield ("Please upload an image.", None, None, None, None, None, None, None, None, None, None)
            return
        base_dir = tempfile.mkdtemp(prefix="foho_outputs_")
        tmp_config = _build_config(
            base_config,
            project_root=REPO_ROOT,
            base_dir=base_dir,
            image_path=image,
            split_path=None,
            conda_sh=conda_sh,
            env_name=os.environ.get("ENV_NAME", "foho"),
            run_inpaint=run_inpaint,
            suppress_warnings=suppress_warnings,
        )
        from foho.configs import load_config
        cfg = load_config(tmp_config)
        img_id = _img_id_from_path(image)

        pipeline_error = {"exc": None}

        def _run():
            try:
                _run_pipeline(tmp_config)
            except Exception as exc:
                pipeline_error["exc"] = exc
            finally:
                if os.path.exists(tmp_config):
                    os.remove(tmp_config)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()

        original = masked_obj = cropped_hoi = cropped_hoi_wo = None
        inpainted = hand_mask = obj_mask = None
        gemini_text = ""
        obj_mesh = hand_mesh = combined_mesh = None

        def _refresh_outputs():
            nonlocal original, masked_obj, cropped_hoi, cropped_hoi_wo
            nonlocal inpainted, hand_mask, obj_mask, gemini_text
            nonlocal obj_mesh, hand_mesh, combined_mesh

            original = original or _find_first(
                os.path.join(cfg.original_img_dir, f"{img_id}_full_image_*.png")
            )
            masked_obj = masked_obj or _find_first(
                os.path.join(cfg.masked_obj_path, f"{img_id}_occ_obj.png")
            )
            cropped_hoi = cropped_hoi or _find_first(
                os.path.join(cfg.cropped_hoi_path, f"{img_id}_cropped_hoi_*.png")
            )
            cropped_hoi_wo = cropped_hoi_wo or _find_first(
                os.path.join(cfg.cropped_hoi_wo_bckg_path, f"{img_id}_cropped_hoi_wo_bckg_*.png")
            )
            inpainted = inpainted or _find_first(
                os.path.join(cfg.cropped_inpainted_obj, f"{img_id}_inpainted_object.png")
            )
            hand_mask = hand_mask or _find_first(
                os.path.join(cfg.mask_dir_path, f"{img_id}_cropped_hand_mask.png")
            )
            obj_mask = obj_mask or _find_first(
                os.path.join(cfg.mask_dir_path, f"{img_id}_cropped_obj_mask.png")
            )

            if cfg.gemini_responses and os.path.isfile(cfg.gemini_responses) and not gemini_text:
                try:
                    import pandas as pd
                    df = pd.read_csv(cfg.gemini_responses)
                    row = df.loc[df["image_id"].astype(str) == str(img_id), "response"]
                    if len(row):
                        gemini_text = str(row.values[0])
                except Exception:
                    pass

            obj_mesh = obj_mesh or _find_first(
                os.path.join(cfg.guidance_out_path, f"{img_id}_obj.ply")
            )
            hand_mesh = hand_mesh or _find_first(
                os.path.join(cfg.guidance_out_path, f"{img_id}_hand.ply")
            )
            if obj_mesh and hand_mesh and not combined_mesh:
                try:
                    import trimesh
                    obj = trimesh.load(obj_mesh, force="mesh")
                    hand = trimesh.load(hand_mesh, force="mesh")
                    combined = trimesh.util.concatenate([obj, hand])
                    combined_path = os.path.join(cfg.guidance_out_path, f"{img_id}_obj_hand.glb")
                    combined.export(combined_path)
                    combined_mesh = combined_path
                except Exception:
                    pass

        while thread.is_alive():
            _refresh_outputs()
            status = "Running… outputs will appear as each step finishes."
            yield (status, original, masked_obj, cropped_hoi, cropped_hoi_wo, inpainted, hand_mask, obj_mask, gemini_text, obj_mesh, hand_mesh, combined_mesh)
            time.sleep(2)

        _refresh_outputs()
        if pipeline_error["exc"] is not None:
            status = f"Run failed: {pipeline_error['exc']}"
        else:
            status = f"Done. Outputs in: {base_dir}"
        yield (status, original, masked_obj, cropped_hoi, cropped_hoi_wo, inpainted, hand_mask, obj_mask, gemini_text, obj_mesh, hand_mesh, combined_mesh)

    with gr.Blocks(title="FollowMyHold Demo") as demo:
        gr.Markdown("# FollowMyHold Demo")
        with gr.Row():
            image = gr.Image(type="filepath", label="Input Image")
        run_inpaint = gr.Checkbox(label="Run Inpaint", value=True)
        suppress_warnings = gr.Checkbox(label="Suppress Warnings", value=True)
        run_btn = gr.Button("Run")
        out = gr.Textbox(label="Status")
        with gr.Row():
            original = gr.Image(label="Original")
            masked_obj = gr.Image(label="Masked Object")
        gemini_out = gr.Textbox(label="Gemini Output")
        with gr.Row():
            hand_mask = gr.Image(label="Hand Mask")
            obj_mask = gr.Image(label="Object Mask")
        with gr.Row():
            cropped_hoi = gr.Image(label="Cropped HOI")
            cropped_hoi_wo = gr.Image(label="Cropped HOI (No BG)")
        with gr.Row():
            inpainted = gr.Image(label="Inpainted Object")
        with gr.Row():
            obj_mesh = gr.Model3D(label="Object Mesh")
            hand_mesh = gr.Model3D(label="Hand Mesh")
            combined_mesh = gr.Model3D(label="Combined Mesh")
        run_btn.click(
            _submit,
            inputs=[image, run_inpaint, suppress_warnings],
            outputs=[out, original, masked_obj, cropped_hoi, cropped_hoi_wo, inpainted, hand_mask, obj_mask, gemini_out, obj_mesh, hand_mesh, combined_mesh],
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode")
    parser.add_argument("--config", help="Base config env file")
    parser.add_argument("--image", help="Input image path")
    parser.add_argument("--split-path", help="Split CSV path")
    parser.add_argument("--base-dir", help="Output base directory (optional)")
    parser.add_argument("--run-inpaint", dest="run_inpaint", action="store_true")
    parser.add_argument("--no-inpaint", dest="run_inpaint", action="store_false")
    parser.set_defaults(run_inpaint=None)
    parser.add_argument("--suppress-warnings", dest="suppress_warnings", action="store_true")
    parser.add_argument("--show-warnings", dest="suppress_warnings", action="store_false")
    parser.set_defaults(suppress_warnings=None)
    args = parser.parse_args()

    if args.cli:
        run_cli(args)
        return

    try:
        demo = build_gradio(args.config)
    except Exception as exc:
        print("Gradio is not available. Re-run with --cli or install gradio.", file=sys.stderr)
        print(f"Import error: {exc}", file=sys.stderr)
        sys.exit(1)

    demo.launch()


if __name__ == "__main__":
    main()
