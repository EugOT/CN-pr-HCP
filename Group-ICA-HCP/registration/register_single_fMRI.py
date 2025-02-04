import os
import sys
import nibabel as nib
import numpy as np
import ants
from joblib import Parallel, delayed
from tqdm import tqdm
import re
import tempfile

def usage():
    print("Usage: python register_single_fMRI.py <input_nifti_file> <output_nifti_file> <template_nifti_file> <num_threads>")
    print("Example: python register_single_fMRI.py input.nii.gz output.nii.gz template.nii.gz 4")
    sys.exit(1)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def main(input_file, output_file, template_file, num_threads):
    img = nib.load(input_file)
    data = img.get_fdata()
    
    print("Calculating mean image...")
    mean_img = np.mean(data, axis=-1)
    mean_img_nii = nib.Nifti1Image(mean_img, img.affine)
    
    with tempfile.TemporaryDirectory() as tempdir:
        mean_img_path = os.path.join(tempdir, 'mean_fMRI_3D.nii.gz')
        nib.save(mean_img_nii, mean_img_path)
        
        fixed = ants.image_read(template_file)
        moving = ants.image_read(mean_img_path)

        print("Registering mean image...")
        registration = ants.registration(
            fixed=fixed,
            moving=moving,
            type_of_transform='antsRegistrationSyNQuick[s]'
        )

        warp_transform = registration['fwdtransforms'][0]
        affine_transform = registration['fwdtransforms'][1]
        
        num_volumes = data.shape[-1]
        print(f"Will register {num_volumes} separate timepoints...")
        
        def process_volume(i):
            volume_data = data[..., i]
            volume_nii = nib.Nifti1Image(volume_data, img.affine)
            temp_path = os.path.join(tempdir, f"timepoint_{i}.nii.gz")
            nib.save(volume_nii, temp_path)

            moving_timepoint = ants.image_read(temp_path)
            registered_timepoint = ants.apply_transforms(
                fixed=fixed, 
                moving=moving_timepoint, 
                transformlist=[warp_transform, affine_transform]
            )
            output_path = os.path.join(tempdir, f"timepoint_{i}_registered.nii.gz")
            ants.image_write(registered_timepoint, output_path)

            return output_path

        registered_files = Parallel(n_jobs=int(num_threads))(delayed(process_volume)(i) for i in tqdm(range(num_volumes)))
        
        print("Done, now merging registered timepoints on temporal axis...")
        
        registered_files.sort(key=natural_sort_key)

        merged_data = []
        for file in registered_files:
            img = nib.load(file)
            merged_data.append(img.get_fdata())

        merged_data = np.stack(merged_data, axis=-1)
        merged_img = nib.Nifti1Image(merged_data, img.affine)
        nib.save(merged_img, output_file)
    
    print("Done! Thanks for the wait!")
    
if __name__ == "__main__":
    
    if len(sys.argv) != 5:
        usage()

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    template_file = sys.argv[3]
    num_threads = sys.argv[4]

    main(input_file, output_file, template_file, num_threads)