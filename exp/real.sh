#PBS -N playground_real
#PBS -S /bin/bash
#PBS -l select=1:ncpus=6:mem=45gb:ngpus=1:host=cvml10

cd ~/attention-interpolation-diffusion
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate AID
python playground.py --model SG161222/RealVisXL_V5.0 --prompt exp/real_new.json
