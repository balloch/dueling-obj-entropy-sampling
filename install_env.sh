<<<<<<< Updated upstream
<<<<<<< Updated upstream
#!/bin/bash
=======
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
conda create -n does python=3.10
conda activate does
pip install -r requirements.txt
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install -U numba
pip install tensorflow_probability==0.23
