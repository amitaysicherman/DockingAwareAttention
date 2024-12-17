#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=128G
#SBATCH --requeue
#SBATCH --gres=gpu:L40:1
#SBATCH -c 8
#SBATCH --array=1-10

configs="--ec_type 0|\
          --ec_type 1|\
          --ec_type 2 --daa_type 1|\
          --ec_type 2 --daa_type 2|\
          --ec_type 2 --daa_type 3|\
          --ec_type 2 --daa_type 4|\
          --ec_type 2 --daa_type 1 --add_ec_tokens 1|\
          --ec_type 2 --daa_type 2 --add_ec_tokens 1|\
          --ec_type 2 --daa_type 3 --add_ec_tokens 1|\
          --ec_type 2 --daa_type 4 --add_ec_tokens 1"

# Split the config string into an array using '|' as a delimiter
IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
python train.py $config