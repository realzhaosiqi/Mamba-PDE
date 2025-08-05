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
#PBS -o output/ns1e-5/simple_withde_out
#PBS -e output/ns1e-5/simple_withde_error

source ~/.bashrc
condapbs_ex LaMO
python3 inr/arom_inr.py --config-name inr_ns1e-5_simple_withde.yaml
