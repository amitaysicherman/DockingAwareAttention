#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:L40:1
#SBATCH --array=7,9,10

#MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))
#echo "MASTER_PORT: $MASTER_PORT"

configs="--ec_type 0|\
     --ec_type 1|\
     --ec_type 2 --daa_type 1|\
     --ec_type 2 --daa_type 2|\
     --ec_type 2 --daa_type 3|\
     --ec_type 2 --daa_type 4|\
     --ec_type 2 --daa_type 1 --add_ec_tokens 1 --esm600m 1|\
     --ec_type 2 --daa_type 2 --add_ec_tokens 1 --esm600m 1|\
     --ec_type 2 --daa_type 3 --add_ec_tokens 1 --esm600m 1|\
     --ec_type 2 --daa_type 4 --add_ec_tokens 1 --esm600m 1"

# Split the config string into an array using '|' as a delimiter
IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
#export PATH=/home/amitay.s/miniconda3/envs/zik/bin/:$PATH
#/home/amitay.s/miniconda3/envs/zik/bin/python -m torch.distributed.launch --nproc_per_node=2 --master_port=$MASTER_PORT train.py $config
#/home/amitay.s/miniconda3/envs/zik/bin/torchrun --nproc_per_node=2 train.py $config
python train.py $config