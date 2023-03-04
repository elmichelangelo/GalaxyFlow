import os
import sys
import pandas as pd
import pickle
import numpy as np
sys.path.append(os.path.dirname(__file__))

def read_file(mainfolder_path):
    dict_result = {
        "loss": [],
        "epochs": [],
        "batch_size": [],
        "generator_random_tensor": [],
        "leakyrelu": [],
        "learning_rate_discriminator":[],
        "learning_rate_generator": [],
        "selected_scaler": []
    }
    cnt = 1
    for trailfolder_name in os.listdir(mainfolder_path):
        trailfolder_path = f"{mainfolder_path}/{trailfolder_name}"
        if os.path.isdir(trailfolder_path) is True:
            try:
                result_file = f"{trailfolder_path}/result.json"
                df_json = pd.read_json(result_file)
                dict_result["loss"].append(df_json["loss"][0])
                dict_result["epochs"].append(df_json["config"]["epochs"])
                dict_result["batch_size"].append(df_json["config"]["batch_size"])
                dict_result["generator_random_tensor"].append(df_json["config"]["generator_random_tensor"])
                dict_result["leakyrelu"].append(df_json["config"]["leakyrelu"])
                dict_result["learning_rate_discriminator"].append(df_json["config"]["learning_rate_discriminator"])
                dict_result["learning_rate_generator"].append(df_json["config"]["learning_rate_generator"])
                dict_result["selected_scaler"].append(df_json["config"]["selected_scaler"])
                # if cnt == 10000:
                #     break
                cnt += 1
            except:
                pass

    df_result = pd.DataFrame(dict_result)

    cond = (df_result['loss'] == df_result["loss"].min())
    result = df_result[cond]
    for key in result.keys():
        print(key, np.array(result[key]))



if __name__ == '__main__':
    read_file(
        mainfolder_path=f"{os.path.abspath(sys.path[0])}/../../Output/ray_results/train_2023-02-07_08-11-19"
    )