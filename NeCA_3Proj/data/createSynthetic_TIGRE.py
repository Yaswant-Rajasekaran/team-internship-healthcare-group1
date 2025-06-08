import tigre
import numpy as np
import tigre.algorithms as algs
import matplotlib.pyplot as plt
import os
import random
import nibabel as nib
from scipy.ndimage import zoom
import pickle

pixel_num=[512, 512]
voxel_num=[128,128,128]
DSD=[993.0, 1055.0]
DSO=[757.06045586378, 756.9934313361]
DDE=[235.93954413621998, 298.0065686639]
pixel_size=[0.278, 0.278]
voxel_size=[0.8, 0.8, 0.8]
first_projection_angle= [29.7, 0.1] 
second_projection_angle= [2, 29] 

def split_coronary_data(file_name):
    img_nifti = nib.load(file_name)
    voxels_space = img_nifti.header['pixdim'][1:4]
    img = img_nifti.get_fdata()
    data = np.array(img)

    data = zoom(data, (voxels_space[0], voxels_space[1], voxels_space[2]), order=0, mode='nearest') > 0
    pos = np.where(data>0.5)
    xyzs = [pos[0], pos[1], pos[2]]

    v_min = np.min(xyzs[0])
    v_max = np.max(xyzs[0])
    xyzs[0] = xyzs[0] - v_min
    x_diff = v_max - v_min

    v_min = np.min(xyzs[1])
    v_max = np.max(xyzs[1])
    xyzs[1] = xyzs[1] - v_min
    y_diff = v_max - v_min

    v_min = np.min(xyzs[2])
    v_max = np.max(xyzs[2])
    xyzs[2] = xyzs[2] - v_min
    z_diff = v_max - v_min

    if x_diff < 128 and y_diff < 128 and z_diff < 128:
        x_gap = 128 - (x_diff + 1)
        y_gap = 128 - (y_diff + 1)
        z_gap = 128 - (z_diff + 1)

        xyzs[0] = xyzs[0] + int(x_gap/2)
        xyzs[1] = xyzs[1] + int(y_gap/2)
        xyzs[2] = xyzs[2] + int(z_gap/2)

        data = np.zeros((voxel_num[0], voxel_num[1], voxel_num[2]))
        data[xyzs[0],xyzs[1],xyzs[2]] = 1

        w, h, d = data.shape
        coords = []
        flag = False
        for i in range(w):
            if flag:
                break
            for j in range(h):
                if flag:
                    break
                for k in range(d):
                    if data[i,j,k] > 0:
                        coords.append([i,j,k])
                        flag = True
                        break

        for [x,y,z] in coords:
            for cx in [x-1,x,x+1]:
                for cy in [y-1,y,y+1]:
                    for cz in [z-1,z,z+1]:
                        c_coord = [cx,cy,cz]
                        if not (c_coord in coords):
                            if cx > -1 and cx < w:
                                if cy > -1 and cy < h:
                                    if cz > -1 and cz < d:
                                        if data[cx,cy,cz] > 0:
                                            coords.append(c_coord)

        coords = np.transpose(np.array(coords))
        data[coords[0], coords[1], coords[2]] = 0
        RCA=data.astype('int8')

        data = data*0
        data[coords[0], coords[1], coords[2]] = 1
        LCA=data.astype('int8')

        return RCA, LCA
    else:
        print('Failed to split for this data!')


def generate_deformed_projections_RCA(phantom):
    # Lets create a geomtry object
    geo = tigre.geometry()
    # Offsets
    geo.offDetector = np.array([0, 0])  # Offset of Detector            (mm)
    # Auxiliary
    geo.accuracy = 1  # Variable to define accuracy of
    geo.COR = 0  # y direction displacement for
    geo.rotDetector = np.array([0, 0, 0])  # Rotation of the detector, by
    geo.mode = "cone"  # Or 'parallel'. Geometry type.
 
    # Detector parameters
    geo.nDetector = np.array(pixel_num)  # number of pixels              (px)
    geo.dDetector = np.array(pixel_size)  # size of each pixel            (mm)
    geo.sDetector = geo.nDetector * geo.dDetector  # total size of the detector    (mm)
    # Image parameters
    geo.nVoxel = np.array([voxel_num[0], voxel_num[1], voxel_num[2]]) # number of voxels              (vx)
    v_size = 90 + 15*np.random.rand()
    geo.sVoxel = np.array([v_size,v_size,v_size]) # total size of the image       (mm)
    geo.dVoxel = np.array(voxel_size)  # size of each voxel            (mm)

    # Distances
    geo.DSD = DSD[0] # Distance Source Detector      (mm)
    geo.DSO = DSO[0] # Distance Source Origin        (mm)
    geo.offOrigin = np.array([0, 0, 0])  # Offset of image from origin   (mm) #detector view:-z,x,y 

    
    angles = np.array([first_projection_angle[0], first_projection_angle[1], 0])
    angles = angles/180*np.pi

    ## Project
    projections_one = tigre.Ax(phantom.copy(), geo, angles) #array
    # projections_one = projections_one > 0

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.imshow(projections[0], cmap=plt.get_cmap('Greys'))
    # plt.show()

    
    ## Reconstruct:
    imgSIRT = algs.sirt(projections_one, geo, angles, 1)
    # imgSIRT_one = imgSIRT > 0
    imgSIRT_one = imgSIRT

    #-8 to 8 mm translation; -10 to 10degrees
    #############################
    # Distances
    geo.DSD = DSD[1] # Distance Source Detector      (mm)
    geo.DSO = DSO[1] # Distance Source Origin        (mm)
    geo.offOrigin = np.array([8*np.random.rand()*random.choice((-1, 1)),8*np.random.rand()*random.choice((-1, 1)),0])

    angles = np.array([second_projection_angle[0], second_projection_angle[1], 0])
    angles = angles/180*np.pi

    ## Project
    projections_two = tigre.Ax(phantom.copy(), geo, angles) #array
    # projections_two = projections_two > 0
    # projections_two = projections_two

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.imshow(projections_two[0], cmap=plt.get_cmap('Greys'))
    # plt.show()
    
    ## Reconstruct:
    geo.offOrigin = np.array([0, 0, 0])
    imgSIRT = algs.sirt(projections_two, geo, angles, 1)
    imgSIRT_two = imgSIRT

    recon = imgSIRT_one.astype(np.int8) + imgSIRT_two.astype(np.int8)

    return projections_one[0], projections_two[0]


filepath='./DeepCA/datasets/CCTA_GT/'
# outpath='./CCTA/'
outpath='./NeCA/data/'
file_names=os.listdir(filepath)
num_data=2
# num_data=len(file_names)
proj_data=np.zeros((num_data, 2, pixel_num[0], pixel_num[1]))

for i, fname in enumerate(file_names):
    # RCA, LCA=split_coronary_data(os.path.join(filepath, fname))

    RCA = np.load(os.path.join(filepath, fname)).astype(np.float32)

    proj_one, proj_two=generate_deformed_projections_RCA(RCA)
    projs=np.array((2, 2, 2))
 
    proj_data[i, 0, :, :]=proj_one[:,:]
    proj_data[i, 1, :, :]=proj_two[:,:]
    print(f'file number {i}')

    if(i==(num_data-1)):
        break


config={'pixel_num':pixel_num,
        'voxel_num': voxel_num,
        'DSD': DSD,
        'DSO': DSO,
        'DDE': DDE,
        'pixel_size': pixel_size,
        'voxel_size': voxel_size,
        'first_projection_angle': first_projection_angle,
        'second_projection_angle': second_projection_angle
        }

data={'projections':proj_data.astype('int8'), 'config':config}

with open(os.path.join(outpath, 'projections.pkl'), 'wb') as f:
    pickle.dump(data, f)
    
# np.save(os.path.join(outpath, 'projections.npy'), data)