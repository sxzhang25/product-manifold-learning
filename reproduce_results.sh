#!/usr/bin/env bash

# VENV='ima' # change this to the name of your virtual environment
# conda activate $VENV

# python factorize.py \
#   --data data/rectangle3d_info.pkl \
#   --configs configs/configs_rectangle3d.json \
#   --outdir results/rectangle3d \
#   --generate_plots

# python factorize.py \
#   --data data/torus_info.pkl \
#   --configs configs/configs_torus.json \
#   --outdir results/torus \
#   --generate_plots

# python factorize.py \
#   --data data/cryo-em_x-theta_noisy_info.pkl \
#   --configs configs/configs_cryo-em_x-theta_noisy.json \
#   --outdir results/cryo-em_x-theta_noisy \
#   --generate_plots

python reproduce_figures.py