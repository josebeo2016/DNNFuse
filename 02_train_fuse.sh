#!/bin/bash
########################
# Script for training
# Usage:
# 
# bash 02_train_category.sh <data_path> <batch_size>
# <data_path>: path to the database
#           e.g. DATA/asvspoof2019
# <batch_size>: depend on your GPU memory, e.g. 10 for 24Gb GPU
#########################

# the random seed
SEED=1234
# the name of the training config file 
CONFIG="configs/conf-1.yaml"
# path to the directory of the model
DATABASE_PATH=$1
# DATABASE_PATH="DATA/asvspoof2019"
# Category model - you should modify this path to your own path if this is not joint training
CATEGORY_MODEL="out/model_60_1_1e-05_conf-1-category/checkpoint_best.pt"
# Comment for the training
CMT="conf-1-fuse" 
# batch size 
BATCH_SIZE=$2
# learning rate should be small if we use logsoftmax as the last layer
LR=0.00001 


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

echo -e "${RED}Training starts${NC}"
echo -e "Training log are writing to $PWD/logs/model_60_1_1e-05_${CMT}"
echo -e "Model save to $PWD/out/model_60_1_1e-05_${CMT}"
com="python main_fuse.py
    --seed ${SEED}
    --config ${CONFIG}
    --database_path ${DATABASE_PATH}
    --category_model ${CATEGORY_MODEL}
    --batch_size ${BATCH_SIZE}
    --comment "${CMT}"
    --num_epochs 60
    --lr ${LR}"

echo ${com}
eval ${com}
echo -e "Training process finished"



