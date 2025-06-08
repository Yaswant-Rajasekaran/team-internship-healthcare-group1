import os
import os.path as osp
import json
import torch
from tqdm import tqdm, trange
from shutil import copyfile
import numpy as np
import random
import math
import yaml
import pickle
import sys
sys.path.append('./NeCA/src')

from dataset import TIGREDataset as Dataset
from network import get_network
from encoder import get_encoder
from render import run_network
# from .render import ct_geometry_projector
from render.ct_geometry_projector import ConeBeam3DProjector
from odl.tomo.util.utility import axis_rotation, rotation_matrix_from_to

filepath='./DeepCA/datasets/CCTA_GT/'
outpath='./NeCA_3proj/data/'

index_path='./DeepCA/outputs_results/data_index.npy'
file_index=np.load(index_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pixel_num=[512, 512]
voxel_num=[128,128,128]
DSD=[993.0, 1055.0]
DSO=[757.06045586378, 756.9934313361]
DDE=[235.93954413621998, 298.0065686639]
pixel_size=[0.278, 0.278]
voxel_size=[0.8, 0.8, 0.8]
first_projection_angle= [29.7, 0.1] 
second_projection_angle= [2, 29] 
third_projection_angle= [90, 29] 

# data={'nDetector':pixel_num,
#       'dDetector':pixel_size,
#       'nVoxel':voxel_num,
#       'dVoxel':voxel_size,
#        'DSD':[DSD],
#        'DSO':[DSO] }

dsd = DSD # Distance Source Detector   mm   
dso = DSO # Distance Source Origin      mm 
dde = DDE

# Detector parameters
proj_size = np.array(pixel_num)  # number of pixels              (px)
proj_reso = np.array(pixel_size) 
# Image parameters
image_size = np.array(voxel_num)  # number of voxels              (vx)
image_reso = np.array(voxel_size)  # size of each voxel            (mm)

first_proj_angle = [-first_projection_angle[1], first_projection_angle[0]]
second_proj_angle = [-second_projection_angle[1], second_projection_angle[0]]
third_proj_angle = [-third_projection_angle[1], third_projection_angle[0]]


def rotation_matrix_to_axis_angle(m):
    angle = np.arccos((m[0,0] + m[1,1] + m[2,2] - 1)/2)

    x = (m[2,1] - m[1,2])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2] - m[2,0])**2 + (m[1,0] -m[0,1])**2)
    y = (m[0,2] - m[2,0])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2]-m[2,0])**2 + (m[1,0]-m[0,1])**2)
    z = (m[1,0] - m[0,1])/math.sqrt((m[2,1]-m[1,2])**2 + (m[0,2]-m[2,0])**2 + (m[1,0]-m[0,1])**2)
    axis=(x,y,z)

    return axis, angle

def generate_deformed_projections_RCA(phantom):
    #### first_projection
    from_source_vec= (0,-dso[0],0)
    from_rot_vec = (-1,0,0)
    to_source_vec = axis_rotation((0,0,1), angle=first_proj_angle[0]/180*np.pi, vectors=from_source_vec)
    to_rot_vec = axis_rotation((0,0,1), angle=first_proj_angle[0]/180*np.pi, vectors=from_rot_vec)
    to_source_vec = axis_rotation(to_rot_vec[0], angle=first_proj_angle[1]/180*np.pi, vectors=to_source_vec[0])

    rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
    proj_axis, proj_angle = rotation_matrix_to_axis_angle(rot_mat)

    ct_projector_first = ConeBeam3DProjector(image_size, image_reso, proj_angle, proj_axis, proj_size, proj_reso, dde[0], dso[0])

    from_source_vec= (0,-dso[1],0)
    from_rot_vec = (-1,0,0)
    to_source_vec = axis_rotation((0,0,1), angle=second_proj_angle[0]/180*np.pi, vectors=from_source_vec)
    to_rot_vec = axis_rotation((0,0,1), angle=second_proj_angle[0]/180*np.pi, vectors=from_rot_vec)
    to_source_vec = axis_rotation(to_rot_vec[0], angle=second_proj_angle[1]/180*np.pi, vectors=to_source_vec[0])

    rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
    proj_axis, proj_angle = rotation_matrix_to_axis_angle(rot_mat)
    
    ct_projector_second = ConeBeam3DProjector(image_size, image_reso, proj_angle, proj_axis, proj_size, proj_reso, dde[1], dso[1])

    from_source_vec= (0,-dso[1],0)
    from_rot_vec = (-1,0,0)
    to_source_vec = axis_rotation((0,0,1), angle=third_proj_angle[0]/180*np.pi, vectors=from_source_vec)
    to_rot_vec = axis_rotation((0,0,1), angle=third_proj_angle[0]/180*np.pi, vectors=from_rot_vec)
    to_source_vec = axis_rotation(to_rot_vec[0], angle=third_proj_angle[1]/180*np.pi, vectors=to_source_vec[0])

    rot_mat = rotation_matrix_from_to(from_source_vec, to_source_vec[0])
    proj_axis, proj_angle = rotation_matrix_to_axis_angle(rot_mat)
    
    ct_projector_third = ConeBeam3DProjector(image_size, image_reso, proj_angle, proj_axis, proj_size, proj_reso, dde[1], dso[1])
    
    train_projs_one = ct_projector_first.forward_project(phantom)
    train_projs_two = ct_projector_second.forward_project(phantom)
    train_projs_three = ct_projector_third.forward_project(phantom)

    proj_one = train_projs_one.cpu().numpy()
    proj_two = train_projs_two.cpu().numpy()
    proj_three = train_projs_three.cpu().numpy()

    return proj_one, proj_two, proj_three

    
num_data=len(file_index)
proj_data=np.zeros((num_data, 3, pixel_num[0], pixel_num[1]))
for i in range(num_data):
    fname=str(file_index[i])+'.npy'
    RCA = np.load(os.path.join(filepath, fname)).astype(np.float32)
    RCA = np.expand_dims(RCA, axis=0)  # shape: (1, 128, 128, 128)
    RCA = np.expand_dims(RCA, axis=0)
    torch_RCA = torch.from_numpy(RCA)
    proj_one, proj_two, proj_three =generate_deformed_projections_RCA(torch_RCA)
    projs=np.array((2, 2, 2))
    proj_data[i, 0, :, :]=proj_one[:,:]
    proj_data[i, 1, :, :]=proj_two[:,:]
    proj_data[i, 2, :, :]=proj_three[:,:]

# for i in range(start_idx, len(file_names)):
#     fname=file_names[i]
    
#     # RCA, LCA=split_coronary_data(os.path.join(filepath, fname))

#     RCA = np.load(os.path.join(filepath, fname)).astype(np.float32)
#     RCA = np.expand_dims(RCA, axis=0)  # shape: (1, 128, 128, 128)
#     RCA = np.expand_dims(RCA, axis=0)
#     torch_RCA = torch.from_numpy(RCA)
#     proj_one, proj_two, proj_three =generate_deformed_projections_RCA(torch_RCA)
#     projs=np.array((2, 2, 2))
#     proj_data[idx, 0, :, :]=proj_one[:,:]
#     proj_data[idx, 1, :, :]=proj_two[:,:]
#     proj_data[idx, 2, :, :]=proj_three[:,:]
    
    
#     print(f'file number {idx}')
    
#     if(idx==(num_data-1)):
#         break

#     idx+=1

## save the results
config={'pixel_num':pixel_num,
        'voxel_num': voxel_num,
        'DSD': DSD,
        'DSO': DSO,
        'DDE': DDE,
        'pixel_size': pixel_size,
        'voxel_size': voxel_size,
        'first_projection_angle': first_projection_angle,
        'second_projection_angle': second_projection_angle, 
        'third_projection_angle' : third_projection_angle
        }

data={'projections':proj_data.astype('int8'), 'config':config, 'file_index':file_index}

with open(os.path.join(outpath, 'projections_odl_test.pkl'), 'wb') as f:
    pickle.dump(data, f)
    