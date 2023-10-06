#!/bin/bash
########################
# Script for training
# Usage:
# 
# bash 02_train_category.sh <data_path> <score_path> <batch_size> <eval_output>
# <data_path>: path to the database
#           e.g. DATA/asvspoof_2019_supcon
# <score_path>: path to the directory of the score
#           e.g. scores/la2019
# <batch_size>: depend on your GPU memory, 
#           e.g. 10 for 24Gb GPU
# <eval_output>: path to the output of the evaluation
#           e.g. docs/la2019_epoch_14.txt
#########################
#CUDA_VISIBLE_DEVICES=3 python main_fuse.py --database_path DATA/df2021 --score_path scores/df2021 --config configs/conf-3_norm.yaml --lr 0.00001 --num_epoch 60 --batch_size 128 --comment "fuse_freeze_conf_3_norm" --category_model out/model_100_10_1e-06_conf-1-category/epoch_15.pth --scoring_model out/model_60_32_1e-05_fuse_freeze_conf_3_norm/epoch_14.pth --eval --eval_output docs/df2021_fuse_freeze_conf_3_norm_epoch_14.txt

# the random seed
SEED=1234
# the name of the training config file 
CONFIG="configs/conf-1.yaml"
# path to the directory of the model
DATABASE_PATH=$1
# DATABASE_PATH="DATA/asvspoof2019"
# score path
SCORE_PATH=$2
# Category model - you should modify this path to your own path if this is not joint training
CATEGORY_MODEL="out/model_60_1_1e-05_conf-1-category/checkpoint_best.pt"
# Scoring model
SCORING_MODEL="out/model_60_1_1e-05_conf-1-fuse/checkpoint_best.pt"
# batch size 
BATCH_SIZE=$2
# eval output
EVAL_OUTPUT=$3


if [ "$#" -ne 2 ]; then
    echo -e "Invalid input arguments. Please check the doc of script."
    exit 1;
fi

# Enter conda environment
eval "$(conda shell.bash hook)"

conda activate fairseq
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "Cannot load fairseq, please run 00_envsetup.sh first"
    exit 1
fi

echo -e "${RED}Evaluation starts${NC}"
com="python main_fuse.py
    --config ${CONFIG}
    --score_path ${SCORE_PATH}
    --database_path ${DATABASE_PATH}
    --category_model ${CATEGORY_MODEL}
    --scoring_model ${SCORING_MODEL}
    --batch_size ${BATCH_SIZE}
    --eval
    --eval_output ${EVAL_OUTPUT}"

echo ${com}
eval ${com}
echo -e "Evaluation process finished"
echo -e "Please calculate the EER by yourself refer to Result.ipynb"

