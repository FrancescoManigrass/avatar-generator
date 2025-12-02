import itertools
import subprocess
import os
from os.path import join
import json

from tqdm import tqdm

PATH = "D:\\Francesco Manigrasso\\avatargenerator\\ScanDB_test_set\\dataDB"
PATH2 = "D:\\Francesco Manigrasso\\avatargenerator\\ScanDB_test_set"

json_measure = {}
f = open(join(PATH2, 'h_w_measures_ScanDB_female.json'))
json_measure["female"] = json.load(f)["female"]

f = open(join(PATH2, 'h_w_measures_ScanDB_male.json'))
json_measure["male"] = json.load(f)["male"]

for gender in os.listdir(PATH):
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(join(PATH, gender)) for f in filenames if
              os.path.splitext(f)[1] == '.png']
    my_dict = {}
    for i in result:
        items = i.split(os.path.sep)
        if items[6] not in my_dict:
            my_dict[items[6]] = {}
        my_dict[items[6]][items[8].replace(".png", "")] = i

    for key, items in tqdm(my_dict.items()):

        current_object= f'subject_mesh_{key}.obj'
        object_name = f'mesh_{int(key)}.obj'
        subprocess.run(f"python demo.py --front \"{items['front']}\" --side \"{items['side']}\" --gender {gender} "
                       f" --height {json_measure[gender][current_object][0]} --weight {json_measure[gender][current_object][1]} --mesh_name {current_object}")
