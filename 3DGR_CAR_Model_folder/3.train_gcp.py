import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

from models.gcp_unet import GCPUNet
from datasets.gcp_dataset import GCPDataset

def get_args():
    parser = argparse.ArgumentParser(description="Train GCP")
    parser.add_argument("--proj_dir", type=str, default="data/processed/projections")
    parser.add_argument("--M_dir", type=str, default="data/processed/M_vectors")
    parser.add_argument("--split_dir", type=str, default="data/splits")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--save_dir", type=str, default="GCP_project/checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def train():
    args = get_args()

    checkpoint_path = Path(args.checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    splits_path = Path(args.splits_dir)
    train_txt = splits_path / "train_list.txt"
    val_txt   = splits_path / "val_list.txt"

    train_dataset = GCPDataset(
        projections_dir=args.proj_dir,
        M_dir=args.M_dir,
        split_txt=train_txt,
        transform=None)

    val_dataset = GCPDataset(
        projections_dir=args.proj_dir,
        M_dir=args.M_dir,
        split_txt=val_txt,
        transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True)

    model = GCPUNet(n_channels=1, n_classes=4, base_filters=32).to(device)
    criterion = nn.L1Loss()   
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0
        
        for batch in train_loader:
            imgs = batch["image"].to(device)  
            M_gt = batch["M"].to(device)      

            min_vals = imgs.amin(dim=[2,3], keepdim=True)
            max_vals = imgs.amax(dim=[2,3], keepdim=True)
            imgs_norm = (imgs - min_vals) / (max_vals - min_vals + 1e-8)

            optimizer.zero_grad()
            output = model(imgs_norm)         
            loss = criterion(output, M_gt)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * imgs.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {avg_train_loss:.6f}")

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(device)
                M_gt = batch["M"].to(device)

                min_vals = imgs.amin(dim=[2,3], keepdim=True)
                max_vals = imgs.amax(dim=[2,3], keepdim=True)
                imgs_norm = (imgs - min_vals) / (max_vals - min_vals + 1e-8)

                preds = model(imgs_norm)
                loss = criterion(preds, M_gt)
                total_val_loss += loss.item() * imgs.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        print(f"Epoch {epoch}/{args.epochs} - Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(args.save_dir, f"gcp_epoch{epoch:03d}_loss{avg_val_loss:.6f}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Updated checkpoint: {ckpt_path}")

    print("Training finished.")

if __name__ == "__main__":
    train()
