import pandas as pd 
import json
import numpy as np
import sys
import os

files = os.listdir("../../Results/json_files/")
sample_file = "results_example.json"
results_path = "../../Results/"
experiments_file = "../../Results/Experiments-list.ods"
sheet_name = "MAML-MMAML"
command_prefix = "python "
command_prefix = "srun python /home/pineda/meta-learning/Code/meta-learning/"
for sheet_name in ["MAML-MMAML", "ABLATION"]:
    
    with open(results_path + "commands_"+sheet_name+".txt", "a+") as f:
        
        vars = ["Experiment index", "Type", "Model", "Training", "Dataset", "Adaptation steps", "Learning rate", "Meta-learning rate", "Noise level", "Task size",  "Vrae weight"]

        if sheet_name=="ABLATION":
            vars += ["ML-Horizon"]
        
        df = pd.read_excel(experiments_file, engine="odf", sheet_name=sheet_name)
        df_selected = df[vars].drop_duplicates()
        df_selected = df_selected[df_selected["Adaptation steps"]!=0]


        for i in range(df_selected.shape[0]):
            
            training = df_selected.iloc[i]["Training"]
            command = command_prefix + ("run_MMAML_04.py" if training =="MMAML" else "run_MAML_04.py ")
            command += " --dataset " + df_selected.iloc[i]["Dataset"]
            command += " --experiment_id " + str(df_selected.iloc[i]["Experiment index"])+"_"+str(df_selected.iloc[i]["Type"])
            command += " --learning_rate " + str(df_selected.iloc[i]["Learning rate"])
            command += " --meta_learning_rate " + str(df_selected.iloc[i]["Meta-learning rate"])
            command += " --noise_level " + str(df_selected.iloc[i]["Noise level"])
            command += " --task_size " + str(df_selected.iloc[i]["Task size"])
            command += " --adaptation_steps " + str(df_selected.iloc[i]["Adaptation steps"])

            if training == "MMAML":
                command += " --weight_vrae " + str(df_selected.iloc[i]["Vrae weight"])

            if sheet_name == "ABLATION":
                command+= " --ml_horizon " + str(df_selected.iloc[i]["ML-Horizon"])

            print(command)
            f.write("%s\n"% command)
