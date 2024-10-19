#PBS -N playground_morph
#PBS -S /bin/bash
#PBS -l select=1:ncpus=6:mem=45gb:ngpus=1:host=cvml03

cd ~/attention-interpolation-diffusion
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate AID
python playground.py --model RunDiffusion/Juggernaut-XL-v9 --prompt exp/ip_morph.json --mode image_to_image
