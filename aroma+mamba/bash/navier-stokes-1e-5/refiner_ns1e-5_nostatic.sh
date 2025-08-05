#!/bin/bash
#PBS -P hn98
#PBS -q gpuvolta
#PBS -l jobfs=50GB
#PBS -l mem=32GB
#PBS -l walltime=48:00:00
#PBS -l storage=scratch/hn98+gdata/hn98
#PBS -l ngpus=1
#PBS -l ncpus=12
#PBS -l wd
#PBS -o output/ns1e-5/refiner_8depth/refiner_nostatic_out
#PBS -e output/ns1e-5/refiner_8depth/refiner_nostatic_error

source ~/.bashrc
condapbs_ex LaMO
python3 dynamics_modeling/train_refiner.py --config-name mamba_ns1e-5_nostatic.yaml
