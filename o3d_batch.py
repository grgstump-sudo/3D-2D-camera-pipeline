from pathlib import Path
import json, gc
import numpy as np
from PIL import Image
from tqdm import tqdm
import open3d as o3d

CFG = {
    "in_dir":  r"C:\Users\Greg\Desktop\Backupply",
    "out_dir": r"C:\Users\Greg\Desktop\pc_renders_alpha",
    "glob": "*.ply",

    "view_json": r"C:\Users\Greg\Desktop\o3d_view_pinhole.json",

    
    "width": 1920,
    "height": 1080,

    "point_size": 1.0,
    "brightness": 1.0,

    "voxel_size": 0.0,
    "max_files": 0,

    "use_resume": False,
    "resume_file": r"C:\Users\Greg\Desktop\o3d_resume_index.txt",
    "gc_every": 10,

    # If your PLYs are slightly translated frame-to-frame and you want the object NOT to drift:
    "translate_to_ref_center": True,

    # Chroma-key fallback for alpha when depth buffer is empty (keeps batch from crashing)
    "chroma_key_fallback": True,
    "chroma_key_rgb": (1.0, 0.0, 1.0),   # magenta
    "chroma_tol": 0.0,                  # tolerance in float space (0..1)
}

def _read_resume():
    p = Path(CFG["resume_file"])
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except Exception:
        return None

def _write_resume(next_index: int):
    try:
        Path(CFG["resume_file"]).write_text(str(next_index), encoding="utf-8")
    except Exception:
        pass

def _load_view():
    p = Path(CFG["view_json"])
    if not p.exists():
        raise SystemExit(f"View JSON not found: {p}\nRun o3d_capture_view_v2.py first.")

    data = json.loads(p.read_text(encoding="utf-8"))

    intr = data["intrinsic"]
    K = np.array(intr["intrinsic_matrix"], dtype=np.float64)
    width = int(intr["width"])
    height = int(intr["height"])
    extr = np.array(data["extrinsic"], dtype=np.float64)

    intrinsic = o3d.camera.PinholeCameraIntrinsic()
    intrinsic.set_intrinsics(width, height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

    params = o3d.camera.PinholeCameraParameters()
    params.intrinsic = intrinsic
    params.extrinsic = extr

    ref_center = None
    if isinstance(data, dict) and "ref_center" in data:
        ref_center = np.array(data["ref_center"], dtype=np.float64)

    return params, (width, height), ref_center

def _ensure_colors(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    if not pcd.has_colors():
        pcd.colors = o3d.utility.Vector3dVector(
            np.ones((len(pcd.points), 3), dtype=np.float64)
        )
    return pcd

def _set_constant_clipping_if_available(ctr):
    # These exist on many Open3D builds; if not, we just skip.
    for name, val in [("set_constant_z_near", 0.01), ("set_constant_z_far", 100000.0)]:
        fn = getattr(ctr, name, None)
        if callable(fn):
            try:
                fn(val)
            except Exception:
                pass

def _pump(vis, n=6):
    for _ in range(int(n)):
        vis.poll_events()
        vis.update_renderer()

def _capture_rgb_depth(vis, settle_frames=6):
    _pump(vis, settle_frames)
    rgb = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    dep = np.asarray(vis.capture_depth_float_buffer(do_render=True))
    rgb8 = (np.clip(rgb, 0, 1) * 255.0).astype(np.uint8)
    alpha = np.where(dep > 0, 255, 0).astype(np.uint8)
    return rgb, rgb8, alpha  # return float rgb too (for chroma tests)

def _chroma_key_alpha(vis, opt, settle_frames=6):
    # Render once with magenta background, then key it out.
    key = np.array(CFG["chroma_key_rgb"], dtype=np.float64)

    # Save + set background
    bg_orig = np.array(opt.background_color, dtype=np.float64)
    opt.background_color = key

    _pump(vis, settle_frames)
    rgb_key = np.asarray(vis.capture_screen_float_buffer(do_render=True))

    # Restore background
    opt.background_color = bg_orig
    _pump(vis, 2)

    # Mask: pixels NOT close to magenta => foreground
    diff = np.abs(rgb_key - key[None, None, :]).max(axis=2)
    mask = diff > float(CFG["chroma_tol"])
    alpha = (mask.astype(np.uint8) * 255)
    return alpha

def main():
    in_dir = Path(CFG["in_dir"])
    out_dir = Path(CFG["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(CFG["glob"]))
    if not files:
        raise SystemExit(f"No PLY files found in {in_dir} with glob {CFG['glob']}")

    base_params, (view_w, view_h), ref_center = _load_view()
    CFG["width"], CFG["height"] = int(view_w), int(view_h)

    start = 0
    if bool(CFG.get("use_resume", False)):
        r = _read_resume()
        if r is not None and r >= 0:
            start = r

    files = files[start:]
    if CFG["max_files"] and int(CFG["max_files"]) > 0:
        files = files[: int(CFG["max_files"])]

    print(f"Rendering {len(files)} files starting at index {start}")
    print(f"Output -> {out_dir}")
    print("IMPORTANT: Leave the Open3D window OPEN and NOT minimized while it runs.")
    print("IMPORTANT: Do NOT resize the Open3D window during rendering (keeps camera stable).")

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        "Open3D Batch Renderer (do not minimize)",
        width=int(CFG["width"]),
        height=int(CFG["height"]),
        visible=True
    )

    opt = vis.get_render_option()
    opt.point_size = float(CFG["point_size"])
    opt.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    opt.light_on = True

    # Keep ONE geometry alive; update it in-place.
    pcd_draw = o3d.geometry.PointCloud()

    # Seed geometry so add_geometry works once.
    seeded = False
    seed_path = None
    for sp in files:
        seed = o3d.io.read_point_cloud(str(sp))
        if seed.is_empty():
            continue
        if CFG["voxel_size"] and float(CFG["voxel_size"]) > 0:
            seed = seed.voxel_down_sample(float(CFG["voxel_size"]))
        seed = _ensure_colors(seed)
        pcd_draw.points = seed.points
        pcd_draw.colors = seed.colors
        seeded = True
        seed_path = sp
        break

    if not seeded:
        raise SystemExit("All PLY files were empty; nothing to render.")

    # KEY CHANGE:
    # We allow ONE reset_bounding_box=True here ONLY to initialize near/far planes.
    # Then we lock the camera params and never reset bbox again.
    vis.add_geometry(pcd_draw, reset_bounding_box=True)

    ctr = vis.get_view_control()

    # Warmup a bit (helps on some Windows/OpenGL setups)
    _pump(vis, 20)

    # Apply saved camera and lock clipping (if available)
    ctr.convert_from_pinhole_camera_parameters(base_params, allow_arbitrary=True)
    _set_constant_clipping_if_available(ctr)
    _pump(vis, 10)

    # Optional: if you want to guarantee the first frame matches your capture PLY:
    # (this doesn’t change view; it just prints what it seeded from)
    if seed_path is not None:
        print(f"Seeded geometry from: {seed_path.name}")

    empty_log = out_dir / "_empty_frames.txt"

    for k, ply in enumerate(tqdm(files), start=0):
        idx = start + k
        out_path = out_dir / f"{ply.stem}.png"
        if out_path.exists():
            _write_resume(idx + 1)
            continue

        pcd = o3d.io.read_point_cloud(str(ply))
        if pcd.is_empty():
            _write_resume(idx + 1)
            continue

        if CFG["voxel_size"] and float(CFG["voxel_size"]) > 0:
            pcd = pcd.voxel_down_sample(float(CFG["voxel_size"]))

        pcd = _ensure_colors(pcd)

        if bool(CFG.get("translate_to_ref_center", False)) and ref_center is not None:
            new_center = np.array(pcd.get_center(), dtype=np.float64)
            pcd.translate(ref_center - new_center, relative=True)

        # Update existing geometry (NO clear/add)
        pcd_draw.points = pcd.points
        pcd_draw.colors = pcd.colors
        vis.update_geometry(pcd_draw)

        # Re-apply exact camera (prevents drift)
        ctr.convert_from_pinhole_camera_parameters(base_params, allow_arbitrary=True)
        _set_constant_clipping_if_available(ctr)

        # Capture (depth-based alpha)
        rgb_float, rgb8, alpha = _capture_rgb_depth(vis, settle_frames=6)

        # If depth is empty, try again with more settle frames
        if alpha.max() == 0:
            rgb_float, rgb8, alpha = _capture_rgb_depth(vis, settle_frames=20)

        # If still empty, fallback to chroma-key alpha (instead of crashing)
        if alpha.max() == 0 and bool(CFG.get("chroma_key_fallback", True)):
            try:
                alpha = _chroma_key_alpha(vis, opt, settle_frames=10)
            except Exception:
                alpha = alpha  # keep zeros

        # If STILL empty: skip + log (don’t die mid-batch)
        if alpha.max() == 0:
            msg = f"EMPTY_FRAME idx={idx} file={ply.name}\n"
            print(msg.strip())
            try:
                with empty_log.open("a", encoding="utf-8") as f:
                    f.write(msg)
            except Exception:
                pass
            _write_resume(idx + 1)
            continue

        if CFG["brightness"] != 1.0:
            rgb8 = np.clip(
                rgb8.astype(np.float32) * float(CFG["brightness"]),
                0, 255
            ).astype(np.uint8)

        rgba = np.dstack([rgb8, alpha])
        Image.fromarray(rgba, mode="RGBA").save(out_path)

        _write_resume(idx + 1)

        del pcd
        if CFG["gc_every"] and ((k + 1) % int(CFG["gc_every"]) == 0):
            gc.collect()

    vis.destroy_window()
    print("Done.")

if __name__ == "__main__":
    main()
