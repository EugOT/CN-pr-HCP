#Dual regression and connectivity map extraction for rs-fMRI.

from conntask_ni import utils
import nibabel as nb
import numpy as np
import pandas as pd
import os
import time
import nilearn.masking
from scipy.signal import detrend
from sklearn.preprocessing import StandardScaler
import warnings
import atexit
from colorama import Fore, Style, init

init()

captured_warnings = []

def display_warnings():
    if captured_warnings:
        print()
        print(Fore.YELLOW + "-"*30 + " WARNING " + "-"*30)
        for warning in captured_warnings:
            print(f"{warning.category.__name__}: {warning.message}")

atexit.register(display_warnings)

##################################################################################
################ SOURCE: https://github.com/ShacharGal/connTask ##################
##################################################################################

def read_nii(entry):
    if isinstance(entry, np.ndarray):
        return entry # the object is already the img
    img = load_nifti(entry)
    return np.asarray(img.get_fdata())

def read_multiple_ts_data(file_paths, trim=None):
    # reads multiple time series files, normalizes, demeans and concatenates
    # trim is used if you do not wish to use all the time points in your data
    all_data = []
    for file_path in file_paths:
        rs = read_nii(file_path).T # original file should be (time,voxels)
        if isinstance(trim, np.ndarray):
            rs = rs[:, trim]
        all_data.append(detrend(StandardScaler().fit_transform(rs)))
    return np.concatenate(all_data, axis=1)

def parse_dr_args(data, components):
    # read data and make sure its in the right dimensions for further processing
    if isinstance(data, list):
        data = utils.read_multiple_ts_data(data)
    elif isinstance(data, str):
        rs = utils.read_data(data).T
        data = detrend(StandardScaler().fit_transform(rs))
    elif isinstance(data, np.ndarray):
        if data.shape[0] < data.shape[1]:
            data = data.T # data should be nvoxels * nica_components
    else:
        warnings.warn('data is supplied in unknown format')

    if isinstance(components, str):
        components = utils.read_data(components).T
    elif isinstance(components, np.ndarray):
        pass
    else:
        print('ica components should be a path to an nii file or an nd.array')
        raise TypeError
    if components.shape[0] < components.shape[1]:
        components = components.T # group ica file should be nvoxels * nica_components
    return data, components

def dual_regression(data, group_components):
    # perform dual regression to yield subject-specific spatial-components
    data, group_components = parse_dr_args(data, group_components)
    # step 1: get components subject-specific time series for each components
    pinv_comp = np.linalg.pinv(group_components)
    ts = np.matmul(pinv_comp, data)
    # step 2: find the subject-specific spatial pattern for each component
    sub_comps = utils.fsl_glm(ts.T, data.T)
    return sub_comps.T

def weighted_seed2voxel(seeds, data):
    # performs a weighted_seed2voxel analysis for each component
    # to yield the connectivity maps used in the connTask prediction pipeline
    pinv_seeds = np.linalg.pinv(seeds)
    ts = np.matmul(pinv_seeds, data)
    ts_norm = utils.normalise_like_matlab(ts.T).T
    data_norm = utils.normalise_like_matlab(data.T).T
    features = np.matmul(ts_norm, data_norm.T).T
    return features

##################################################################################
##################################################################################
##################################################################################

def load_nifti(path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        file = nb.load(path)
        captured_warnings.extend(w)
        return file

def fit_mask(mask_file_nifti, img):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        masked_pixels = nilearn.masking.apply_mask(img, mask_img=mask_file_nifti)
        print(f'\tShape of masked image: {masked_pixels.shape}')
        captured_warnings.extend(w)
        return masked_pixels

from argparse import ArgumentParser

parser = ArgumentParser(description='Script used to perform dual regression and connectivity map regression for resting state fMRI. Output can be used as input to SwiFUN. Group ICA file is required to be produced by MELODICA and masked beforehand.')
parser.add_argument('--groupICA_dir', default='registered_input', type=str, help='Assume that the group ICA file is registered and flattened (dim: # component * # voxels except backgrounds), needs to be masked beforehand. Assumed to be in directory "output" and called "melodic_IC.nii.gz".')
parser.add_argument('--outdir', default='out-features', type=str)
parser.add_argument('--start_idx', default=0, type=int)
parser.add_argument('--maskfile', default='metadata/mask_IC.nii.gz', type=str,help='Use the mask created by create_masks.py.')
parser.add_argument('--rs_data_dir', default='rs_data', type=str, help='Directory containing the resting state data from which the features should be extracted.')
parser.add_argument('--rs_output_file', default='features_42_comps', type=str,help='Name suffix of the output file for the resting state features.')

args = parser.parse_args()

if not os.path.exists(args.outdir):
    create_dir = input("Output directory does not exist. Do you want to create it? (y/n): ")
    if create_dir.lower() == "y":
        os.makedirs(args.outdir)
        print(Fore.GREEN + f"Output directory '{args.outdir}' created.")
    else:
        print(Fore.RED + "Output directory does not exist and user chose not to create it. Exiting...")
        exit()

# set an output directory for the features
outdir = args.outdir

# set origin directory for raw data
rs_data_dir = args.rs_data_dir

print('Loading mask...')
try:
    mask_file = load_nifti(args.maskfile)
    print(f"Shape of mask: {mask_file.get_fdata().astype(int).shape}")
    mask = mask_file
except Exception as e:
    print(Fore.RED + f'Could not load mask: {e}' + Style.RESET_ALL)
    exit()

print(Fore.GREEN + f'Successfully loaded mask {args.maskfile}!', Style.RESET_ALL)

print('Loading group ICA...')
try:
    group_ica_file = np.load(os.path.join(args.groupICA_dir, "output", "melodic_IC_masked.npy"))
    print(f"Shape of group ICA map: {group_ica_file.shape}")
    group_ica = group_ica_file
except Exception as e:
    print(Fore.RED + f'Could not load group ICA map: {e}' + Style.RESET_ALL)
    exit()
        
print(Fore.GREEN + 'Successfully loaded and masked group ICA map.', Style.RESET_ALL)

# iterate through all participants and perform feature extraction
subjects = [ subj for subj in os.listdir(rs_data_dir) ]

for i, sub in enumerate(sorted(subjects)[args.start_idx:]):
    rs_file = f'{rs_data_dir}/{sub}'
    print(Style.RESET_ALL + f'subject {i+1}/{len(subjects)}: {sub}')
    start=time.time()
    try:
        if not os.path.exists(f'{outdir}/{sub}_{args.rs_output_file}.npy'): 
            print(f'\tLoading image...')
            rs_image = load_nifti(rs_file)
            
            # extract subject ID and session ID from the file name
            file_name = os.path.basename(rs_file)
            subject_id = file_name.split('.')[0]
            
            masked_pixels = fit_mask(mask, rs_image)

            rs_paths = [masked_pixels]

            data = read_multiple_ts_data(rs_paths)

            # perform two steps of feature extraction
            print('\tExtracting features...')
            dr_comps = dual_regression(data, group_ica)
            features = weighted_seed2voxel(dr_comps, data)

            print('\tSaving features...')
            to_save = features.T
            np.save(f'{outdir}/{subject_id}_{args.rs_output_file}', to_save)
            print(Fore.GREEN + f'\tSuccess! Features saved to {outdir}/{subject_id}_{args.rs_output_file}.npy' + Style.RESET_ALL)
        else:
            print('\tFile already exists!')
    except Exception as e:
        print(Fore.RED + f'\tHaving problems in file {sub}...')
        print(Fore.RED + f'\tFollowing error occured: {e}' + Style.RESET_ALL)
        with open('corrupted_files.txt','a') as f:
            f.write(f'{sub}\n')
    finally:
        end=time.time()
        print(f'\tTime taken to process {sub[:15]}: {end-start}')
