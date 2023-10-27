#!/bin/bash
#PBS -N LLAMA_lora
#PBS -l select=1:ncpus=10:mem=10gb
#PBS -l walltime=1:00:00
#PBS -j oe
#PBS -P personal-e0134107
#PBS -o log/evaluation.log

# The following environment variables will be preset after your job is submitted and started.
################################################# 
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
#################################################


source activate lora


cd $PBS_O_WORKDIR
# cd alpaca_finetuning_v1
[ -d log ] || mkdir log  



python evaluation.py