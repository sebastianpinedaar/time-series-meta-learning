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

vars = ["Experiment index", "Type", "Model", "Training", "Dataset", "Adaptation steps", "Learning rate", "Meta-learning rate", "Noise level", "Task size",  "Vrae weight"]


df_selected = df[vars].drop_duplicates()
df_selected = df_selected[df_selected["Adaptation steps"]!=0]

print(df_selected)


for sheet_name in ["MAML-MMAML", "ABLATION"]:

    with open(results_path + "commands_"+sheet_name+".txt", "a+") as f:
        
        df = pd.read_excel(experiments_file, engine="odf", sheet_name=sheet_name)

        for i in range(df_selected.shape[0]):
            
            command = command_prefix + ("run_MMAML_04.py" if df_selected.iloc[i]["Training"] =="MMAML" else "run_MAML_04.py ")
            command += " --dataset " + df_selected.iloc[i]["Dataset"]
            command += " --type " + df_selected.iloc[i]["Type"]
            command += " --index " + str(df_selected.iloc[i]["Experiment index"])
            command += " --learning_rate " + str(df_selected.iloc[i]["Learning rate"])
            command += " --meta_learning_rate " + str(df_selected.iloc[i]["Meta-learning rate"])
            command += " --noise_level " + str(df_selected.iloc[i]["Noise level"])
            command += " --task_size " + str(df_selected.iloc[i]["Task size"])
            command += " --vrae_weight " + str(df_selected.iloc[i]["Vrae weight"])

            print(command)
            f.write("%s\n"% command)
