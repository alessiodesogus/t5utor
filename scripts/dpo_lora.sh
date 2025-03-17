#!/bin/bash -l
#SBATCH --chdir /scratch/izar/reategui/
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 4
#SBATCH --mem 64G
#SBATCH --time 48:00:00
#SBATCH --gpus=1
#SBATCH --account cs-552
#SBATCH --qos cs-552

module load gcc python
source ~/venvs/project/bin/activate
echo "Starting training"

python ~/project-m2-2024-mnlpredators/scripts/dpo_lora.py --model_name MBZUAI/LaMini-Flan-T5-783M --dataset M4-ai/prm_dpo_pairs --batch_size 2 --gradient_accumulation_steps 16 --test_size 0.05 --logging_steps 50 --eval_steps 250 --save_steps 500 --num_epochs 3