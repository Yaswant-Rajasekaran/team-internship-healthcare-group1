import os
import torch
import numpy as np
from pathlib import Path

from models.gcp_unet import GCPUNet
from models.gaussian_reconstructor import GaussianReconstructor

CHECKPOINT_PATH = "GCP_project/checkpoints/gcp_epoch050_loss3.837418.pth"
PROJ_DIR = "data/processed/projections"
RESULTS_DIR = "results/3dgr_volumes"

STEP = 16     
DEPTH_THR = 0.05     
NUM_ITER = 1000     
LR = 1e-3     
SIGMA_0 = 1.0      
INTENSITY_0 = 0.1    

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def m_to_coords(M_pred, view: str, step: int, depth_thr: float):
    
    device = M_pred.device
    
    d_map, dx_map, dy_map, dz_map = M_pred[0], M_pred[1], M_pred[2], M_pred[3]
    coords = []

    for row in range(0, 128, step):
        for col in range(0, 128, step):
            d_val = float(d_map[row, col])
            if d_val <= depth_thr:
                continue

            if view == "angle0":
                x = d_val * 127.0
                y = row + float(dy_map[row, col]) * 10.0
                z = col + float(dz_map[row, col]) * 2.0
            else:  
                y = d_val * 127.0
                x = row + float(dx_map[row, col]) * 10.0
                z = col + float(dz_map[row, col]) * 2.0

            if 0 <= x < 128 and 0 <= y < 128 and 0 <= z < 128:
                coords.append([x, y, z])

    if not coords:
        return torch.empty((0, 3), device=device)
    
    return torch.tensor(coords, device=device, dtype=torch.float32)

def reconstruct_volume():
    out_dir = Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    device = torch.device(out_dir)

    model = GCPUNet(n_channels=1, n_classes=4, base_filters=32).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    volume_id = "1"
    proj_0 = os.path.join(PROJ_DIR, f"{volume_id}_angle0.npy")
    proj_90 = os.path.join(PROJ_DIR, f"{volume_id}_angle90.npy")

    if not (os.path.isfile(proj_0) and os.path.isfile(proj_90)):
        print(f"Error: projections not found.")
        return

    proj0 = torch.from_numpy(np.load(proj_0).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    proj90 = torch.from_numpy(np.load(proj_90).astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)

    for tensor in (proj0, proj_90):
        min = tensor.amin((2,3), keepdim=True)
        max = tensor.amax((2,3), keepdim=True)
        tensor.sub_(min).div_(max - min + 1e-8)

    with torch.no_grad():
        out0 = model(proj0).squeeze(0)
        out90 = model(proj90).squeeze(0)

    pts0 = m_to_coords(out0, "angle0", STEP, DEPTH_THR)
    pts90 = m_to_coords(out90, "angle90", STEP, DEPTH_THR)
    init_points = torch.cat([pts0, pts90], dim=0)

    if init_points.numel() == 0:
        print("No valid points found.")
        init_points = torch.tensor([[64.0, 64.0, 64.0]], device=device)

    
    skel0_path = os.path.join(PROJ_DIR, f"{volume_id}_angle0_skel.npy")
    skel90_path = os.path.join(PROJ_DIR, f"{volume_id}_angle90_skel.npy")

    if os.path.isfile(skel0_path) and os.path.isfile(skel90_path):
        skel0 = torch.from_numpy(np.load(skel0_path).astype(np.float32)).to(device)
        skel90 = torch.from_numpy(np.load(skel90_path).astype(np.float32)).to(device)
    else:
        skel0 = torch.ones((128,128), device=device)
        skel90 = torch.ones((128,128), device=device)

    drr0 = proj0[0,0]
    drr90 = proj90[0,0]

    reconstructor = GaussianReconstructor(
        init_mu = init_points,
        init_sigma = SIGMA_0,
        init_intensity = INTENSITY_0,
        device = device).to(device)

    print(f"Starting Optimization: num_iter={NUM_ITER}, lr={LR}, stride={STEP}")
    reconstructor.optimize(
        real_projs = [drr0, drr90],
        real_masks = [skel0, skel90],
        angles = [0, 90],
        num_iter = NUM_ITER,
        lr = LR)

    with torch.no_grad():
        vol_3d = reconstructor.forward().cpu().squeeze().numpy()  

    vol_bin = (vol_3d > 0.1).astype(np.uint8)              

    out_cont = os.path.join(RESULTS_DIR, "1_3dgr_continuous.npy")
    out_bin = os.path.join(RESULTS_DIR, "1_3dgr_binary.npy")
    np.save(out_cont, vol_3d)
    np.save(out_bin,  vol_bin)
    print("Reconstruction completed")

if __name__ == "__main__":
    reconstruct_volume()
