#!/bin/bash
conda env create -n dreamerv3 -f dreamerv3jax.yml
conda activate dreamerv3
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

