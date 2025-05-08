#!/bin/bash
#SBATCH --job-name=word_embeddings
#SBATCH --output=/home/users/%u/out/word_embeddings_%j.out
#SBATCH --error=/home/users/%u/err/word_embeddings_%j.err
#SBATCH --time=15:00
#SBATCH --partition=normal
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

mkdir -p /scratch/users/$USER/nltk_data
export NLTK_DATA=/scratch/users/$USER/nltk_data  # Define custom NLTK data path

module load python/3.12  #Load Python

python3 /farmshare/learning/CESTA25/batchHug.py #Run script
