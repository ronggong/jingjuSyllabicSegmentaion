#!/bin/bash

#SBATCH -J joint_syl_sub
#SBATCH -p high
#SBATCH -N 1
#SBATCH --workdir=/homedtic/rgong/cnnSyllableSeg/jingjuSyllabicSegmentation
#SBATCH --gres=gpu:maxwell:1
#SBATCH --sockets-per-node=1
#SBATCH --cores-per-socket=2
#SBATCH --threads-per-core=2

# Output/Error Text
# ----------------
#SBATCH -o /homedtic/rgong/cnnSyllableSeg/out/joint_syl.%N.%J.%u.out # STDOUT
#SBATCH -e /homedtic/rgong/cnnSyllableSeg/out/joint_syl.%N.%J.%u.err # STDERR

# anaconda environment
export PATH=/homedtic/rgong/anaconda2/bin:$PATH
source activate /homedtic/rgong/keras_env

python /homedtic/rgong/cnnSyllableSeg/jingjuSyllabicSegmentation/training_scripts/newHpcDLScriptsTest/keras_cnn_syllableSeg_jan_madmom_original_deep_new_hpc.py