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


print(files)
if sample_file not in files:

  with open('../../Results/json_files/results_example.json', 'w') as outfile:


    experiments_file = pd.read_excel(experiments_file, engine="odf", sheet_name="MAML-MMAML")
    print("Fields:", experiments_file.columns)

    data = {}
    for field in experiments_file.columns:
      x = str(experiments_file[field][0])
      data[field] = x if not pd.isna(x) else "NAN"

    json.dump([data], outfile)

else:

  files.pop(files.index(sample_file))

  df = pd.read_excel(experiments_file, engine="odf", sheet_name=sheet_name)

  if sheet_name == "MAML-MMAML":

    for file in files:

      with open('../../Results/json_files/'+file) as json_file:
      
        json_data = json.load(json_file)
     


        for data in json_data:
          print(data)
          data["Experiment index"] = data["Experiment_id"].split("_")[0]
          data["Type"] = data["Experiment_id"].split("_")[1]
          
          #if data["Type"]!= "COMPARISON":
          #  continue

          df.loc[(df["Experiment index"].astype(str) == data["Experiment index"]) & 
                (df["Type"].astype(str) == data["Type"]) &
                (df["Model"].astype(str) == data["Model"])&
                (df["Training"].astype(str) == data["Training"]) &
                (df["Dataset"].astype(str) == data["Dataset"]) &
                (df["Adaptation steps"] == data["Adaptation steps"]) &
                (df["Meta-learning rate"] == data["Meta-learning rate"]) &
                (df["Learning rate"] == data["Learning rate"]) &
                (df["Noise level"] == data["Noise level"]) &
                (df["Task size"]== data["Task size"]) &
                (df["Horizon"] == data["Horizon"]) &
                (df["Evaluation loss"].astype(str) == data["Evaluation loss"]) & 
                (df["Vrae weight"]== data["Vrae weight"]) &
                (df["Trial"] == data["Trial"]) ,[ "Value"]]= data["Value"]

        df.to_excel(results_path+"Experiments_results_"+sheet_name+".xlsx")

  if sheet_name == "ABLATION":

    for file in files:

      with open('../../Results/json_files/'+file) as json_file:
          
        json_data = json.load(json_file)

        for data in json_data:
          
          data["Experiment index"] = data["Experiment_id"].split("_")[0]
          data["Type"] = data["Experiment_id"].split("_")[1]
          
          if data["Type"]!= "ABLATION":
            continue

          df.loc[(df["Experiment index"].astype(str) == data["Experiment index"]) & 
                (df["Type"].astype(str) == data["Type"]) &
                (df["Model"].astype(str) == data["Model"])&
                (df["Training"].astype(str) == data["Training"]) &
                (df["Dataset"].astype(str) == data["Dataset"]) &
                (df["Adaptation steps"] == data["Adaptation steps"]) &
                (df["Meta-learning rate"] == data["Meta-learning rate"]) &
                (df["Learning rate"] == data["Learning rate"]) &
                (df["Noise level"] == data["Noise level"]) &
                (df["Task size"] == data["Task size"]) &
                (df["Horizon"] == data["Horizon"]) &
                (df["Evaluation loss"].astype(str) == data["Evaluation loss"]) & 
                (df["Vrae weight"]== data["Vrae weight"]) &
                (df["Trial"] == data["Trial"]) & 
                (df["ML-Horizon"]== data["ML-Horizon"]) ,[ "Value"]]= data["Value"]

        df.to_excel(results_path+"Experiments_results_"+sheet_name+".xlsx")


