import os
import random
from pathlib import Path

root = Path("data/processed")
M_dir   = root / "M_vectors"     

all_M = sorted(f for f in os.listdir(M_dir) if f.endswith("_M.npy"))

samples = [fname[:-len("_M.npy")] for fname in all_M] 

random.seed(42)
random.shuffle(samples)

split_idx = int(0.8 * len(samples))
train_set = samples[:split_idx]
val_set = samples[split_idx:]

split_dir = Path("data/splits")
split_dir.mkdir(parents=True, exist_ok=True)

with open(split_dir, "train_list.txt", "w") as f_train:
    for i in train_set:
        f_train.write(i + "\n")

with open(split_dir, "val_list.txt", "w") as f_val:
    for item in val_set:
        f_val.write(item + "\n")
