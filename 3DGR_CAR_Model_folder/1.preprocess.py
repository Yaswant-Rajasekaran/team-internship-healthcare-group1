import os
import numpy as np
import nibabel as nib
from scipy.ndimage import rotate, zoom
from skimage.morphology import skeletonize_3d, skeletonize
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt

def resize_volume(vol: np.ndarray, dim=(128, 128, 128)) -> np.ndarray:
    factors = [t / s for t, s in zip(dim, vol.shape)]
    return zoom(vol, zoom=factors, order=1)

def projections(vol: np.ndarray, angle: float) -> np.ndarray:
    min, max = vol.min(), vol.max()
    vol_norm = (vol - min) / (max - min + 1e-8)
    vol_rot = rotate(vol_norm, angle=angle, axes=(1, 0), reshape=False, order=1)
    if angle % 180 == 0:
        projection = np.sum(vol_rot, axis=0)  
    else:
        projection = np.sum(vol_rot, axis=1)  
    projection = (projection - projection.min()) / (projection.max() - projection.min() + 1e-8)
    if projection.shape != (128, 128):
        projection = zoom(projection, (128 / projection.shape[0], 128 / projection.shape[1]), order=1)
    return projection.astype(np.float32)

def depth_map(vol: np.ndarray, angle: float, depth_thr=0.5) -> np.ndarray:
    vol_bin = (vol > depth_thr).astype(np.uint8)
    if angle % 180 != 0:
        vol_bin = vol_bin.transpose(0, 2, 1)
    depth_map = np.argmax(vol_bin, axis=0).astype(np.int32)
    no_voxel_mask = ~np.any(vol_bin, axis = 0)
    depth_map[no_voxel_mask] = 128
    return depth_map 


def build_M_for_view(depth_map: np.ndarray, skeleton_pts: np.ndarray) -> np.ndarray:
    H, W = depth_map.shape
    M = np.zeros((4, H, W), dtype=np.float32)

    M[0] = depth_map.astype(np.float32) / 128.0

    max_depth = depth_map.max() 
    mask = depth_map < max_depth

    if skeleton_pts.size == 0 or not mask.any():
        return M
    
    ys, xs = np.nonzero(mask)
    zs = depth_map[ys, xs]

    query_pts = np.column_stack((ys, xs, zs))

    skel_yxz = skeleton_pts[:, [1, 0, 2]]

    tree = cKDTree(skel_yxz)
    _, idxs = tree.query(query_pts)
    nearest = skel_yxz[idxs]  

    offsets = nearest - query_pts  
    dx = offsets[:, 1] / 10.0
    dy = offsets[:, 0] / 10.0
    dz = offsets[:, 2] / 2.0

    M[1][ys, xs] = dx
    M[2][ys, xs] = dy
    M[3][ys, xs] = dz

    return M

def extract_skeleton_3d(seg: np.ndarray) -> np.ndarray:
    skel = skeletonize_3d(seg.astype(np.uint8))
    zs, ys, xs = np.nonzero(skel)
    return np.column_stack((xs, ys, zs))


def process_all(data_dir, output_root, limit=200, angles=[0, 90], depth_threshold=0.5):
    subdirs = ("projections", "skeletons3d", "skeletons2d", "M_vectors" )
    for d in subdirs: 
        path = os.path.join(output_root, d)
        os.makedirs(path, exist_ok=True)

    for idx in range(1, limit + 1):
        base_name = str(idx)
        img_path  = os.path.join(data_dir, f"{base_name}.img.nii.gz")
        lbl_path  = os.path.join(data_dir, f"{base_name}.label.nii.gz")
        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            print(f"[Warning] {base_name} not found.")
            continue

        vol = nib.load(img_path).get_fdata().astype(np.float32)
        seg = nib.load(lbl_path).get_fdata().astype(np.uint8)

        vol128 = resize_volume(vol, (128, 128, 128))
        seg128 = resize_volume(seg, (128, 128, 128))
        seg128 = (seg128 > 0.5).astype(np.uint8)  

        pts3d = extract_skeleton_3d(seg128)
        np.save(os.path.join(output_root, "skeletons3d", f"{base_name}_ske3d.npy"), pts3d)

        for angle in angles:
            drr = projections(vol128, angle_deg=angle)
            out_drr_npy = os.path.join(output_root, "projections", f"{base_name}_angle{angle}.npy")
            out_drr_png = os.path.join(output_root, "projections", f"{base_name}_angle{angle}.png")
            np.save(out_drr_npy, drr)
            plt.imsave(out_drr_png, drr, cmap="gray")

            drr_bin = (drr > 0.5).astype(np.uint8)
            ske2d = skeletonize(drr_bin)
            out_ske2d = os.path.join(output_root, "skeletons2d" f"{base_name}_angle{angle}_skel.npy")
            np.save(out_ske2d, ske2d.astype(np.float32))

            dp = depth_map(vol128, angle_deg=angle, depth_threshold=depth_threshold)
            M = build_M_for_view(dp, pts3d)
            out_M = os.path.join(output_root, "M_vectors", f"{base_name}_angle{angle}_M.npy")
            np.save(out_M, M)

            print(f"[Processed] ID {base_name}, angle {angle}: DRR, 2D-skel e M saved.")
    print("Process completed")
    
if __name__ == "__main__":
    DATA_DIR      = "data/1-200"       
    OUTPUT_ROOT   = "data/processed"   
    LIMIT_VOLUMES = 200
    ANGLES        = [0, 90]
    DEPTH_THRESH  = 0.5               

    process_all(
        data_dir=DATA_DIR,
        output_root=OUTPUT_ROOT,
        limit=LIMIT_VOLUMES,
        angles=ANGLES,
        depth_threshold=DEPTH_THRESH
    )
