#!/bin/bash

# change python version
module load cuda/8.0
#module load theano/0.8.2

# two variables you need to set
device=gpu1  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
#
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=0.95
export LD_LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/soft/cuda/cudnn/cuda/include:$CPATH
export LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH

source activate /homedtic/rgong/keras_env

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongcnnSyllableSeg_node07 ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_node07
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcnnSyllableSeg_node07
mkdir /scratch/rgongcnnSyllableSeg_node07/syllableSeg
mkdir /scratch/rgongcnnSyllableSeg_node07/error
mkdir /scratch/rgongcnnSyllableSeg_node07/script
mkdir /scratch/rgongcnnSyllableSeg_node07/out

# Thir, copy the experiment's data:
# ----------------------------------
cp -rp /homedtic/rgong/cnnSyllableSeg/syllableSeg/* /scratch/rgongcnnSyllableSeg_node07/syllableSeg

#$ -N cnnSyllableSeg
#$ -q default.q
#$ -l h=node01

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/cnnSyllableSeg/out/cnn.$JOB_ID.out
#$ -e /homedtic/rgong/cnnSyllableSeg/error/cnn.$JOB_ID.err

python /homedtic/rgong/cnnSyllableSeg/keras_cnn_syllableSeg_conv_dense_horizontal_timbral_filters_jordi.py

# Copy data back, if any
# ----------------------
printf "rgongcnnSyllableSeg_node07 processing done. Moving data back\n"
cp -rf /scratch/rgongcnnSyllableSeg_node07/out/* /homedtic/rgong/cnnSyllableSeg/out
printf "_________________\n"
#
# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcnnSyllableSeg_node07 ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_node07
fi
printf "Job done. Ending at `date`\n"
