import json
import os
from os.path import join

import numpy as np
from PIL import Image
from tqdm import tqdm

PATH = "D:\\Francesco Manigrasso\\avatargenerator\\Dataset"
result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if os.path.splitext(f)[1] == '.npy' if "train_test_data_fold1" in os.path.join(dp, f) and 'images'  in os.path.join(dp, f)]



for i in tqdm(result):

    data = np.load(i)
    current_json_path = "D:\\Francesco Manigrasso\\avatargenerator\\Dataset\\data10"+"\\"+i.split("\\")[5]+".json"

    f = open(current_json_path)
    json_data = json.load(f)

    gender= i.split("\\")[7]
    dataset_type= i.split("\\")[-1].split("_")[0]
    for j in range(len(data)):
        rows = i.replace("Dataset\\","Dataset\\images\\").split("\\")

        path = rows[:-1]
        path.append(rows[-1].replace(".npy",""))
        output_path = "\\".join(path)
        os.makedirs(output_path,exist_ok=True)
        front = Image.fromarray(data[j][:,:,0])
        front.save(join(output_path,"front_"+json_data[gender][dataset_type][j]+".jpg"))
        back = Image.fromarray(data[j][:,:,1])
        front.save(join(output_path,"back_"+json_data[gender][dataset_type][j]+".jpg"))



