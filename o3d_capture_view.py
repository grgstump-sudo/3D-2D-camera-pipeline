from pathlib import Path
import json
import numpy as np
import open3d as o3d

CFG = {
    "ply_path": r"C:\Users\Greg\Desktop\Backupply\cloud_000001.ply",
    "width": 1920,
    "height": 1080,
    "point_size": 0.002,
    "out_view_json": r"C:\Users\Greg\Desktop\o3d_view_pinhole.json",
}

def _to_list(mat):
    return np.asarray(mat).tolist()

def main():
    ply = Path(CFG["ply_path"])
    if not ply.exists():
        raise SystemExit(f"PLY not found: {ply}")

    pcd = o3d.io.read_point_cloud(str(ply))
    if pcd.is_empty():
        raise SystemExit("Point cloud is empty.")

    if not pcd.has_colors():
        pcd.colors = o3d.utility.Vector3dVector(
            np.ones((len(pcd.points), 3), dtype=np.float64)
        )

    # Reference center (helps optional stabilization/alignment in the batch renderer)
    ref_center = np.array(pcd.get_center(), dtype=np.float64)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(
        "Adjust view then press S to save (Q to quit)",
        width=int(CFG["width"]),
        height=int(CFG["height"]),
        visible=True
    )
    vis.add_geometry(pcd, reset_bounding_box=True)

    opt = vis.get_render_option()
    opt.point_size = float(CFG["point_size"])
    opt.background_color = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    opt.light_on = True

    def save_view(_vis):
        ctr = _vis.get_view_control()
        params = ctr.convert_to_pinhole_camera_parameters()
        data = {
            "capture_ply": str(ply),
            "ref_center": ref_center.tolist(),
            "intrinsic": {
                "width": int(params.intrinsic.width),
                "height": int(params.intrinsic.height),
                "intrinsic_matrix": _to_list(params.intrinsic.intrinsic_matrix),
            },
            "extrinsic": _to_list(params.extrinsic),
        }
        outp = Path(CFG["out_view_json"])
        outp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"[Saved] {outp}")
        return False  # keep running

    def quit_app(_vis):
        print("Quitting.")
        _vis.close()
        return True

    vis.register_key_callback(ord("S"), save_view)
    vis.register_key_callback(ord("Q"), quit_app)

    print("INSTRUCTIONS:")
    print(" - Rotate/zoom the view until it matches your desired angle.")
    print(" - Press S to save viewpoint to:", CFG["out_view_json"])
    print(" - Press Q to quit.")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()
