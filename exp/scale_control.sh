#PBS -N playground_scale_control
#PBS -S /bin/bash
#PBS -l select=1:ncpus=6:mem=45gb:ngpus=1:host=cvml03

cd ~/attention-interpolation-diffusion
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate AID
python playground.py --model RunDiffusion/Juggernaut-XL-v9 --prompt exp/scale_control.json --mode text_to_image
