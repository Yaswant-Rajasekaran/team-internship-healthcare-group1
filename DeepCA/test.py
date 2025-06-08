import numpy as np
import torch
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
ab_path = os.getcwd()
filepath = ab_path + '/outputs_results/checkpoint.pth'
checkpoint = torch.load(filepath, map_location=device)

train_index=checkpoint['train_index']
validation_index=checkpoint['validation_index']
test_index=checkpoint['test_index']

# test_idx=test_index.to('cpu').numpy()

outpath=ab_path+'/outputs_results/data_index.npy'
np.save(outpath, test_index)