"""Helper script executed within Blender to render a single frame with GPU."""

import argparse
import json
import sys
from pathlib import Path

import bpy


def _configure_cycles(device: str, compute_device: str) -> None:
    cycles_addon = bpy.context.preferences.addons.get("cycles")
    if not cycles_addon:
        raise RuntimeError("Cycles add-on not available in this Blender build")

    prefs = cycles_addon.preferences
    prefs.get_devices()
    compute_device = compute_device.upper()
    prefs.compute_device_type = compute_device
    for device_item in prefs.devices:
        if device_item.type == compute_device:
            device_item.use = True
        elif device_item.type == "CPU":
            device_item.use = device.upper() != "GPU"
        else:
            device_item.use = False

    scene = bpy.context.scene
    scene.cycles.device = "GPU" if device.upper() == "GPU" else "CPU"


def _apply_render_settings(args: argparse.Namespace) -> None:
    scene = bpy.context.scene
    if args.samples is not None:
        scene.cycles.samples = args.samples
    if args.resolution_x is not None:
        scene.render.resolution_x = args.resolution_x
    if args.resolution_y is not None:
        scene.render.resolution_y = args.resolution_y
    if args.resolution_percentage is not None:
        scene.render.resolution_percentage = args.resolution_percentage
    if args.file_format is not None:
        scene.render.image_settings.file_format = args.file_format
    if args.color_mode is not None:
        scene.render.image_settings.color_mode = args.color_mode
    if args.color_depth is not None:
        scene.render.image_settings.color_depth = args.color_depth
    if args.use_adaptive_sampling is not None:
        scene.cycles.use_adaptive_sampling = bool(int(args.use_adaptive_sampling))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(output_dir / args.output_basename)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Execute a single Cycles render")
    parser.add_argument("--frame", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-basename", required=True)
    parser.add_argument("--device", default="GPU")
    parser.add_argument("--compute-device", default="CUDA")
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--resolution-x", type=int, default=None)
    parser.add_argument("--resolution-y", type=int, default=None)
    parser.add_argument("--resolution-percentage", type=int, default=None)
    parser.add_argument("--file-format", default=None)
    parser.add_argument("--color-mode", default=None)
    parser.add_argument("--color-depth", default=None)
    parser.add_argument("--use-adaptive-sampling", default=None)

    args = parser.parse_args(argv)

    bpy.context.scene.frame_set(args.frame)
    _configure_cycles(args.device, args.compute_device)
    _apply_render_settings(args)

    bpy.ops.render.render(write_still=True)

    output_path = bpy.path.abspath(bpy.context.scene.render.frame_path(frame=args.frame))
    print(json.dumps({"output_path": output_path}))


if __name__ == "__main__":
    if "--" in sys.argv:
        arg_list = sys.argv[sys.argv.index("--") + 1 :]
    else:
        arg_list = sys.argv[1:]
    main(arg_list)
