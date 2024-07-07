#!/bin/bash
cd ..
git clone https://github.com/balloch/realworldrl_suite.git
git clone https://github.com/eilab-gt/NovGrid.git
cd NovGrid
git checkout new_envs
cd ..
#git clone https://github.com/balloch/rl-exploration-transfer.git
#cd rl-exploration-transfer
#git checkout sweeps_and_analysis
#conda create -n dm_transfer python=3.8 -y
#conda activate dm_transfer
export MUJOCO_GL=egl # glfw
#conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
conda install cmake bzip2 -y
cd dueling-obj-entropy-sampling
python -m pip install -r env_req.txt
python -m pip install ../realworldrl_suite
#python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
python -m pip install shimmy
python -m pip install gymnasium
python -m pip install stable_baselines3
python -m pip install minigrid
python -m pip install wandb
python -m pip install tensorboard
python -m pip install tqdm
python -m pip install -e ../NovGrid
python -m pip install imageio
