#!/bin/bash
#SBATCH --job-name=full_experiment
#SBATCH --output=full_experiment_out.txt
#SBATCH --error=full_experiment_err.txt
#SBATCH -p sched_mit_sloan_batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3-00:00:00
#SBATCH --constraint="centos7"
#SBATCH $HOME/anaconda3/bin/python


xvfb-run -d python3 full_experiment.py
