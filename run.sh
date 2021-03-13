#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=0
#SBATCH --mem=0
#SBATCH --time=0-01:00:00
#SBATCH --account=def-lorenzf
#SBATCH --output=out.txt

echo 'hello'
source ~/ENV/bin/activate
python src/main.py
