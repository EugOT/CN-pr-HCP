#!/bin/bash

#SBATCH -A m4727_g # project to charge for this job
#SBATCH -N 1 # number of nodes
#SBATCH -C gpu # type of architecture (gpu or cpu)
#SBATCH -q regular # quality of service (regular, debug, ...)
#SBATCH -J swift-test # name of job
#SBATCH -t 10:00:00 # max walltime
#SBATCH --ntasks-per-node=1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none

cd /pscratch/sd/s/styllp/intro_neuro/SwiFT
#source /usr/anaconda3/etc/profile.d/conda.sh
module load conda
conda activate py39

# Check if PyTorch can access the GPUs
python -c "import torch; print(torch.cuda.is_available())"

echo "Available environments:"
conda env list

TRAINER_ARGS='--accelerator gpu --max_epochs 10 --precision 16 --num_nodes 1 --devices 1 --strategy DDP' # specify the number of gpus as '--devices'
MAIN_ARGS='--loggername neptune --classifier_module v6 --dataset_name S1200 --image_path /pscratch/sd/j/junbeom/HCP_filtered_run1_MNI_to_TRs'
DATA_ARGS='--batch_size 8 --num_workers 8 --input_type rest'
DEFAULT_ARGS='--project_name SwiFT-dHCP-sex'
OPTIONAL_ARGS='--c_multiplier 2 --last_layer_full_MSA True --clf_head_version v1 --downstream_task age --downstream_task_type regression' #--use_scheduler --gamma 0.5 --cycle 0.5'
RESUME_ARGS=''

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkYjdlMTEyZS00NGNhLTQyZDctYTg4OC0wMGU4NWMyZjMwNzIifQ==" # when using neptune as a logger

#export CUDA_VISIBLE_DEVICES=0
 
python project/main.py $TRAINER_ARGS $MAIN_ARGS $DEFAULT_ARGS $DATA_ARGS $OPTIONAL_ARGS $RESUME_ARGS \
--dataset_split_num 1 --seed 1 --learning_rate 5e-5 --model swin4d_ver7 --depth 2 2 6 2 --embed_dim 36 \
--sequence_length 20 --first_window_size 2 2 2 2 --window_size 4 4 4 4 --img_size 96 96 96 20 \
--patch_size 6 6 6 1 # --augment_during_training
#--load_model_path /global/cfs/cdirs/m4750/dowon/v2_pretrained_model/SWIF4-235/checkpt-epoch=39-valid_loss=0.12.ckpt 
#--mask_patch_size 6 6 6 4 --mask_ratio 0.8
#--load_model_path /pscratch/sd/s/styllp/infant-fmri/pretrained_models/contrastive_pretrained.ckpt
