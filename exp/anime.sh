#PBS -N playground_anime
#PBS -S /bin/bash
#PBS -l select=1:ncpus=6:mem=45gb:ngpus=1:host=cvml10

cd ~/attention-interpolation-diffusion
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate AID
python playground.py --model cagliostrolab/animagine-xl-3.1 --prompt exp/anime_new.json
