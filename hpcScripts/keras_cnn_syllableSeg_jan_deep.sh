#!/bin/bash

# change python version
#module load cuda/8.0
#module load theano/0.8.2

# two variables you need to set
device=gpu0  # the device to be used. set it to "cpu" if you don't have GPUs

# export environment variables
#
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
export THEANO_FLAGS=mode=FAST_RUN,device=$device,floatX=float32,lib.cnmem=0.95
export PATH=/usr/local/cuda/bin/:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH
export CPATH=/soft/cuda/cudnn/cuda/include:$CPATH
export LIBRARY_PATH=/soft/cuda/cudnn/cuda/lib64:$LD_LIBRARY_PATH

source activate /homedtic/rgong/keras_env

printf "Removing local scratch directories if exist...\n"
if [ -d /scratch/rgongcnnSyllableSeg_jan_deep ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_jan_deep
fi

# Second, replicate the structure of the experiment's folder:
# -----------------------------------------------------------
mkdir /scratch/rgongcnnSyllableSeg_jan_deep
mkdir /scratch/rgongcnnSyllableSeg_jan_deep/syllableSeg
mkdir /scratch/rgongcnnSyllableSeg_jan_deep/error
mkdir /scratch/rgongcnnSyllableSeg_jan_deep/script
mkdir /scratch/rgongcnnSyllableSeg_jan_deep/out

# Thir, copy the experiment's data:
# ----------------------------------
cp -rp /homedtic/rgong/cnnSyllableSeg/syllableSeg/train_set_all_syllableSeg_mfccBands2D.pickle.gz /scratch/rgongcnnSyllableSeg_jan_deep/syllableSeg
cp -rp /homedtic/rgong/cnnSyllableSeg/syllableSeg/sample_weights_syllableSeg_mfccBands2D.pickle.gz /scratch/rgongcnnSyllableSeg_jan_deep/syllableSeg

#$ -N cnnS_old_d
#$ -q default.q
#$ -l h=node05

# Output/Error Text
# ----------------
#$ -o /homedtic/rgong/cnnSyllableSeg/out/cnn_jan_old_deep.$JOB_ID.out
#$ -e /homedtic/rgong/cnnSyllableSeg/error/cnn_jan_old_deep.$JOB_ID.err

python /homedtic/rgong/jingjuSyllableSegmentation/hpcScripts/keras_cnn_syllableSeg_jan_deep.py

# Copy data back, if any
# ----------------------
printf "rgongcnnSyllableSeg_jan_deep processing done. Moving data back\n"
cp -rf /scratch/rgongcnnSyllableSeg_jan_deep/out/* /homedtic/rgong/cnnSyllableSeg/out
printf "_________________\n"
#
# Clean the crap:
# ---------------
printf "Removing local scratch directories...\n"
if [ -d /scratch/rgongcnnSyllableSeg_jan_deep ]; then
        rm -Rf /scratch/rgongcnnSyllableSeg_jan_deep
fi
printf "Job done. Ending at `date`\n"
