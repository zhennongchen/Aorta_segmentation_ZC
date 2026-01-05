import numpy as np
import glob 
import os
from PIL import Image
import math
import SimpleITK as sitk
import cv2
import random
import nibabel as nb
from scipy.ndimage import binary_erosion, distance_transform_edt


def load_nrrd(file_path):
    image_file = sitk.ReadImage(file_path)
    spacing = np.array(image_file.GetSpacing())
    origin = np.array(image_file.GetOrigin()) 
    direction = np.array(image_file.GetDirection()).reshape(3, 3)
    affine = np.eye(4)
    affine[:3, :3] = direction * spacing.reshape(1, 3)
    affine[:3, 3] = origin
    data = sitk.GetArrayFromImage(image_file)
    return data,affine, spacing, origin, direction
        
    
def nrrd_to_nii_orientation(nrrd_data, format = 'nrrd'):
    if format[0:4] == 'nrrd':
        nrrd_data = np.rollaxis(nrrd_data,0,3)
    return np.rollaxis(np.flip(np.rollaxis(np.flip(nrrd_data, axis=0), -2, 2), axis = 2),1,0)

def create_lowpass_mask(shape, radius):
    H, W, S = shape
    center = np.array([H // 2, W // 2, S // 2])
    y, x, z = np.ogrid[:H, :W, :S]  # 顺序和 shape 对应
    dist = np.sqrt((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2)
    mask = dist <= radius
    return mask

def create_2d_lowpass_mask(shape, radius):
    H, W = shape
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    mask = dist <= radius
    return mask


def pick_random_from_segments(X):
    # Generate the list from 0 to X
    full_list = list(range(X + 1))
    
    # Determine the segment size
    segment_size = len(full_list) // 4

    # Initialize selected numbers
    selected_numbers = []

    # Loop through each segment and randomly pick one number
    for i in range(4):
        start = i * segment_size
        end = (i + 1) * segment_size if i < 3 else len(full_list)  # Ensure last segment captures all remaining elements
        segment = full_list[start:end]
        selected_numbers.append(random.choice(segment))

    return selected_numbers


# function: set window level
def set_window(image,level,width):
    if len(image.shape) == 3:
        image = image.reshape(image.shape[0],image.shape[1])
    new = np.copy(image)
    high = level + width // 2
    low = level - width // 2
    # normalize
    unit = (1-0) / (width)
    new[new>high] = high
    new[new<low] = low
    new = (new - low) * unit 
    return new

# function: get first X numbers
# if we have 1000 numbers, how to get the X number of every interval numbers?
def get_X_numbers_in_interval(total_number, start_number, end_number , interval = 100):
    '''if no random pick, then random_pick = [False,0]; else, random_pick = [True, X]'''
    n = []
    for i in range(0, total_number, interval):
        n += [i + a for a in range(start_number,end_number)]
    n = np.asarray(n)
    return n


# function: find all files under the name * in the main folder, put them into a file list
def find_all_target_files(target_file_name,main_folder):
    F = np.array([])
    for i in target_file_name:
        f = np.array(sorted(glob.glob(os.path.join(main_folder, os.path.normpath(i)))))
        F = np.concatenate((F,f))
    return F

# function: find time frame of a file
def find_timeframe(file,num_of_dots,start_signal = '/',end_signal = '.'):
    k = list(file)

    if num_of_dots == 0: 
        num = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk = k[num+1:]
    
    else:
        if num_of_dots == 1: #.png
            num1 = [i for i, e in enumerate(k) if e == end_signal][-1]
        elif num_of_dots == 2: #.nii.gz
            num1 = [i for i, e in enumerate(k) if e == end_signal][-2]
        num2 = [i for i,e in enumerate(k) if e== start_signal][-1]
        kk=k[num2+1:num1]


    total = 0
    for i in range(0,len(kk)):
        total += int(kk[i]) * (10 ** (len(kk) - 1 -i))
    return total

# function: sort files based on their time frames
def sort_timeframe(files,num_of_dots,start_signal = '/',end_signal = '.'):
    time=[]
    time_s=[]
    
    for i in files:
        a = find_timeframe(i,num_of_dots,start_signal,end_signal)
        time.append(a)
        time_s.append(a)
    time_s.sort()
    new_files=[]
    for i in range(0,len(time_s)):
        j = time.index(time_s[i])
        new_files.append(files[j])
    new_files = np.asarray(new_files)
    return new_files

# function: make folders
def make_folder(folder_list):
    for i in folder_list:
        os.makedirs(i,exist_ok = True)


# function: save grayscale image
def save_grayscale_image(a,save_path,normalize = True, WL = 50, WW = 100):
    I = np.zeros((a.shape[0],a.shape[1],3))
    # normalize
    if normalize == True:
        a = set_window(a, WL, WW)

    for i in range(0,3):
        I[:,:,i] = a
    
    Image.fromarray((I*255).astype('uint8')).save(save_path)


# function: normalize translation control points:
def convert_translation_control_points(t, dim, from_pixel_to_1 = True):
    if from_pixel_to_1 == True: # convert to a space -1 ~ 1
        t = [tt / dim * 2 for tt in t]
    else: # backwards
        t = [tt / 2 * dim for tt in t]
    
    return np.asarray(t)



# function: dice
def np_categorical_dice(pred, truth, k):
    """ Dice overlap metric for label k """
    A = (pred == k).astype(np.float32)
    B = (truth == k).astype(np.float32)
    return 2 * np.sum(A * B) / (np.sum(A) + np.sum(B))


# function: erode and dilate
def erode_and_dilate(img_binary, kernel_size, erode = None, dilate = None):
    img_binary = img_binary.astype(np.uint8)

    kernel = np.ones(kernel_size, np.uint8)  

    if dilate is True:
        img_binary = cv2.dilate(img_binary, kernel, iterations = 1)

    if erode is True:
        img_binary = cv2.erode(img_binary, kernel, iterations = 1)
    return img_binary


def np_categorical_dice(pred, truth, target_class, exclude_class = None):
    if exclude_class is not None:
        valid_mask = (truth != exclude_class)
        pred = pred * valid_mask
        truth = truth * valid_mask

    """ Dice overlap metric for label k """
    A = (pred == target_class).astype(np.float32)
    B = (truth == target_class).astype(np.float32)
    return (2 * np.sum(A * B) + 1e-8) / (np.sum(A) + np.sum(B) + 1e-8)

def HD_95_numpy(pred,gt,spacing):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    # surface voxels
    pred_surf = pred ^ binary_erosion(pred)
    gt_surf = gt ^ binary_erosion(gt)

    dt_gt = distance_transform_edt(~gt_surf, sampling=spacing)
    dt_pred = distance_transform_edt(~pred_surf, sampling=spacing)

    d_pred_gt = dt_gt[pred_surf]
    d_gt_pred = dt_pred[gt_surf]

    hd95=max(np.percentile(d_pred_gt,95), np.percentile(d_gt_pred,95))

    return hd95