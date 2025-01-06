import matplotlib.pyplot as plt
import numpy as np
from skimage import measure
import skimage as ski
import math

# Load the 3D Image as a raw binary file
file_path = "26_QS_PD_a_A_nlm_tv_cor_reg_mrf_fill.raw"
# Input the image dimension (z,y,x). Grain = 1, Pore space = 0
image = np.fromfile(file_path, dtype=np.uint8).reshape((576, 751, 751))

pc_values_list = [1000, 2000, 3000] #Pa

saturation_values = []
pc_values = np.array(pc_values_list) #Pa

mask = np.ones_like(image, dtype='uint8')

for pc in pc_values:
    resolution = 8.42 #micron
    sigma = 26 #mN/m
    kernel_size = round((((2*sigma*(math.cos(math.radians(0))))/pc)*1000/resolution))
    ball = ski.morphology.ball(kernel_size, dtype=np.uint8)
    kernel_size_nwp = round((((2*sigma)/pc)*1000/resolution))
    ball_nwp = ski.morphology.ball(kernel_size_nwp, dtype=np.uint8)

    grain_dilation = ski.morphology.binary_dilation(image, footprint=ball)

    NW_reservoir = np.zeros((1, 751, 751), dtype=np.uint8)
    ready_pore_labels = np.concatenate((NW_reservoir, grain_dilation), axis=0)

    pore_labels = measure.label(ready_pore_labels, background=1)

    value_connected_pores = np.unique(pore_labels[0,:,:])

    pore_labels[(pore_labels!=0)&(pore_labels!=value_connected_pores)] = 10
    pore_labels[pore_labels==value_connected_pores]= 127
    pore_labels[pore_labels==0] = 2
    pore_labels[pore_labels==10] = 1
    pore_labels[pore_labels==127] = 0

    pore_labels[pore_labels == 2] = 1

    ready_for_nwd = pore_labels[1:573, :, :]

    #nwp_dilation = ski.morphology.binary_erosion(ready_for_nwd, footprint=ball)

    ready_for_nwd_slices = []

    for i in range(ready_for_nwd.shape[0]):
        if 0 in ready_for_nwd[i, :, :] and 1 in ready_for_nwd[i, :, :]:
            ready_for_nwd_slices.append(ready_for_nwd[i, :, :])

    ready_for_nwd_slices = np.array(ready_for_nwd_slices)

    slices_count = ready_for_nwd_slices.shape [0]

    if ready_for_nwd_slices.shape[0] == 0:
        nwp_dilation = ready_for_nwd

    elif ready_for_nwd_slices.shape[0] < ready_for_nwd.shape[0]:
        pre_nwp_dilation=ski.morphology.binary_erosion(ready_for_nwd_slices, footprint=ball_nwp)
        nwp_dilation = np.copy(ready_for_nwd)
        nwp_dilation[:slices_count, :, :][pre_nwp_dilation == 0] = 0

    else:
        nwp_dilation = ski.morphology.binary_erosion(ready_for_nwd, footprint=ball_nwp)

    comb = np.copy(image)
    comb[comb==0]=3
    comb[comb==1]=2
    comb[comb==3]=1
    comb[nwp_dilation==0] = 0
    comb[image==1] = 2
    
    comb[mask==0]=1

    wp_reservoir = np.ones((1, 751, 751), dtype=np.uint8)
    ready_pore_labels_2 = np.concatenate((comb, wp_reservoir), axis=0)
    ready_pore_labels_2 [ready_pore_labels_2==0] = 2
    pore_labels_2 = measure.label(ready_pore_labels_2, background=2)

    value_connected_pores_2 = np.unique(pore_labels_2[572,:,:])

    a = pore_labels_2[:-1, :, :]

    trapped = np.ones_like(image, dtype='uint8')
    trapped[(a!=0) & (a!=value_connected_pores_2)]=0
    mask = trapped

    wp = np.sum(comb==1)
    nwp =np.sum(comb==0)
    saturation = (1-(nwp/(wp+nwp))) *100

    saturation_values.append(saturation)

    comb.tofile(f'saturation_pc_{pc}.raw')

    print(f"pc: {pc}, Saturation: {saturation}%")

pc_values = pc_values[:len(saturation_values)]
plt.plot(saturation_values, pc_values, marker='o')
plt.xlabel('Saturation (%)')
plt.ylabel('Pc')
plt.title('Saturation vs. pc')
plt.grid()
plt.show()
