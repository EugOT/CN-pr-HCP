#!/bin/bash

# This script provides a way to register fMRI timepoints to a template image.

set -e

for cmd in fslmaths antsRegistrationSyNQuick.sh antsApplyTransforms fslroi fslmerge fslval parallel; do
    if ! command -v $cmd &> /dev/null; then
        echo "Error: $cmd could not be found. Please install it before running this script."
        exit 1
    fi
done

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <input_nifti_file> <output_nifti_file> <template_nifti_file> <num_threads>"
    echo "Example: $0 input.nii.gz output.nii.gz week40_T1w_215mm.nii.gz 4"
    exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"
TEMPLATE_FILE="$3"
NUM_THREADS="$4"

echo "Generating a mean 3D image..."
fslmaths "$INPUT_FILE" -Tmean mean_fMRI_3D.nii.gz
if [ $? -ne 0 ]; then
    echo "Error: fslmaths command failed."
    exit 1
fi

echo "Registering mean image..."
antsRegistrationSyNQuick.sh -d 3 -f "$TEMPLATE_FILE" -m mean_fMRI_3D.nii.gz -o output_prefix_
if [ $? -ne 0 ]; then
    echo "Error: antsRegistrationSyNQuick.sh command failed."
    exit 1
fi

mkdir -p registered_timepoints
if [ $? -ne 0 ]; then
    echo "Error: Failed to create directory registered_timepoints."
    exit 1
fi

num_volumes=$(fslval "$INPUT_FILE" dim4)
if [ $? -ne 0 ]; then
    echo "Error: fslval command failed."
    exit 1
fi

echo "Will register $num_volumes timepoints..."

process_volume() {
    i="$1"
    INPUT_FILE="$2"
    TEMPLATE_FILE="$3"

    fslroi "$INPUT_FILE" "registered_timepoints/timepoint_${i}.nii.gz" "$i" 1
    if [ $? -ne 0 ]; then
        echo "Error: fslroi command failed for timepoint $i."
        exit 1
    fi

    antsApplyTransforms -d 3 -i "registered_timepoints/timepoint_${i}.nii.gz" \
        -r "$TEMPLATE_FILE" -t output_prefix_1Warp.nii.gz -t output_prefix_0GenericAffine.mat \
        -o "registered_timepoints/timepoint_${i}_registered.nii.gz"
    if [ $? -ne 0 ]; then
        echo "Error: antsApplyTransforms command failed for timepoint $i."
        exit 1
    fi

	rm "registered_timepoints/timepoint_${i}.nii.gz" 

    echo "Timepoint $i finished..."
}

export -f process_volume

parallel -j "$NUM_THREADS" process_volume ::: $(seq 0 1 $(($num_volumes - 1))) ::: "$INPUT_FILE" ::: "$TEMPLATE_FILE"

echo "Finished! Now merging files..."
fslmerge -t "$OUTPUT_FILE" $(ls registered_timepoints/timepoint_*_registered.nii.gz | sort -V)
if [ $? -ne 0 ]; then
    echo "Error: fslmerge command failed."
    exit 1
fi

echo "All done, now let me clean up a bit..."
rm -rdf registered_timepoints
rm -f output_prefix_*
rm -f mean_fMRI_3D.nii.gz

echo "Done! Thanks for the wait!"

