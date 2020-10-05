#!/usr/bin/env bash
#SBATCH --job-name=MAMLPOLL
#SBATCH --output=Logs/MAMLPOLL-%j.log
#SBATCH --error=Logs/MAMLPOLL-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=pineda@uni-hildesheim.de


# ## FOR GPU USE:
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
source /home/pineda/miniconda3/bin/activate pytorch12

export  HDF5_USE_FILE_LOCKING=FALSE

## Run the script
##srun python /home/pineda/meta-learning/Code/eval_base_models.py --model LSTM --dataset BATTERY --lower_trial 0 --upper_trial 3 --learning_rate 0.01 --mode WOFT --save_model_file hyp_model.pt
srun python /home/pineda/meta-learning/Code/meta-learning/run_MAML_04.py --lower_trial 0 --upper_trial 3 --adaptation_steps 5 --meta_learning_rate 0.0005 --learning_rate 0.001 --epochs 1000 --batch_size 20 --stopping_patience 20
srun python /home/pineda/meta-learning/Code/meta-learning/run_MAML_04.py --lower_trial 0 --upper_trial 3 --adaptation_steps 5 --meta_learning_rate 0.005 --learning_rate 0.01 --epochs 1000 --batch_size 20 --stopping_patience 20
srun python /home/pineda/meta-learning/Code/meta-learning/run_MAML_04.py --lower_trial 0 --upper_trial 3 --adaptation_steps 5 --meta_learning_rate 0.00005 --learning_rate 0.0001 --epochs 1000 --batch_size 20 --stopping_patience 20
