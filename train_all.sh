#!/bin/bash
#SBATCH --time=7-00
#SBATCH --mem=256G
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -w bruno[1-3],euler1,houdini,newton5,nlp-l40-[1-2],tdk-bm4,dym-lab[1-2],galileo[1-2],newton[3-4],nlp-a40-1,chuck[1-2] 
#SBATCH --array=1-3

#MASTER_PORT=$((29500 + SLURM_ARRAY_TASK_ID))
#echo "MASTER_PORT: $MASTER_PORT"

#configs="--ec_type 0|\
#     --ec_type 1|\
#     --ec_type 2 --daa_type 1|\
#     --ec_type 2 --daa_type 2|\
#     --ec_type 2 --daa_type 3|\
#     --ec_type 2 --daa_type 4|\
#     --ec_type 2 --daa_type 1 --add_ec_tokens 1|\
#     --ec_type 2 --daa_type 2 --add_ec_tokens 1|\
#     --ec_type 2 --daa_type 3 --add_ec_tokens 1|\
#     --ec_type 2 --daa_type 4 --add_ec_tokens 1"
configs=" --ec_type 2 --daa_type 1 --add_ec_tokens 1 --emb_suf _pb1|\
 --ec_type 2 --daa_type 1 --add_ec_tokens 1 --emb_suf _gn1|\
  --ec_type 2 --daa_type 1 --add_ec_tokens 1 --emb_suf _re"

# Split the config string into an array using '|' as a delimiter
IFS='|' read -ra config_array <<< "$configs"
config=${config_array[$((SLURM_ARRAY_TASK_ID - 1))]}
python train.py $config