#!/bin/bash

# change python version
module load cuda/7.5
#module load theano/0.8.2

# two variables you need to set
device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
#
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=0.95
export LD_LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/soft/cuda/cudnn/cuda/include:$CPATH
export LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH

source activate /homedtic/rgong/keras_env


#$ -N sseg_tem
#$ -q default.q
#$ -l h=node05

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/cnnSyllableSeg/out/cnn_temporal.$JOB_ID.out
#$ -e /homedtic/rgong/cnnSyllableSeg/error/cnn_temporal.$JOB_ID.err

python /homedtic/rgong/cnnSyllableSeg/keras_cnn_syllableSeg_conv_dense_jordi.py

printf "Job done. Ending at `date`\n"
