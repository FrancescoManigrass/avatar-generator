"""Questo script ha lo scopo di calcolare la distanza di Hausdorff tra le mesh di due cartelle diverse,
che contengono le mesh da confrontare indicizzate allo stesso modo.

Innanzitutto, viene definita una funzione chiamata hausdorff_distance, che prende in input due file OBJ
e restituisce la distanza di Hausdorff tra le due mesh.

La funzione carica i file OBJ come oggetti Trimesh e calcola tutte le distanze tra i punti delle due mesh,
poi calcola le distanze di Hausdorff, che vengono restituite come output.

Nella funzione main(), viene letto un file JSON chiamato "height_weight.json" e vengono prese solo
le informazioni relative ai soggetti di genere femminile.
Successivamente, viene eseguita una demo per ogni soggetto femminile del test set.

Per ogni soggetto, vengono definite le immagini di input (front e side) e il nome della mesh in output.

Successivamente, viene eseguito un comando che lancia una demo di un modello che prende in input le immagini front e side,
e restituisce la mesh in formato OBJ. Questa mesh viene salvata nella cartella specificata dal comando.

Dopo aver generato tutte le nuove mesh, viene calcolata la distanza di Hausdorff tra le mesh nelle due cartelle (folder1 e folder2).
Per ogni mesh nella cartella 1, viene controllato se esiste un file corrispondente nella cartella 2.
Se il file esiste, la distanza di Hausdorff tra le due mesh viene calcolata utilizzando la funzione hausdorff_distance.

Infine, viene stampato il risultato della distanza di Hausdorff per ogni coppia di mesh corrispondente."""
import concurrent
import multiprocessing
import random
from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import csv
import threading

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import numpy as np
import trimesh
import argparse
from joblib import dump, load
import pandas as pd
from tqdm import tqdm
import tqdm.notebook as tq
import numpy as np
import trimesh
import os

from body_measurements.measurement import Body3D


def calculate_measures(mesh1, mesh2):
    o_body = Body3D(mesh1.vertices, mesh1.faces)
    r_body = Body3D(mesh2.vertices, mesh2.faces)

    o_height = o_body.height()
    r_height = r_body.height()
    height = np.abs(o_height - r_height)

    o_weight = o_body.weight()
    r_weight = r_body.weight()
    weight = np.abs(o_weight - r_weight)

    _, chest_location, o_chest = o_body.chest()
    _, chest_location, r_chest = r_body.chest()
    chest = np.abs(o_chest - r_chest)

    _, waist_location, o_waist = o_body.waist()
    _, waist_location, r_waist = r_body.waist()
    waist = np.abs(o_waist - r_waist)

    _, hip_location, o_hip = o_body.hip()
    _, hip_location, r_hip = r_body.hip()
    hip = np.abs(o_hip - r_hip)

    return height, weight, chest, waist, hip


def hausdorff_distance(file1, file2):
    # Carica i file OBJ come oggetti Trimesh
    mesh1 = file1
    mesh2 = file2

    # Calcola tutte le distanze tra i punti delle due mesh
    dists1 = np.sqrt(np.sum(np.square(mesh1.vertices[:, np.newaxis] - mesh2.vertices), axis=2))
    dists2 = np.sqrt(np.sum(np.square(mesh2.vertices[:, np.newaxis] - mesh1.vertices), axis=2))

    # Calcola le distanze di Hausdorff
    hausdorff_dist = np.max(np.array([np.max(np.min(dists1, axis=0)), np.max(np.min(dists2, axis=0))]))

    return hausdorff_dist


class Human():
    def __init__(self, kernel, alpha, degree, shape_dirs, template, faces):
        self.kernel = kernel
        self.alpha = alpha
        self.degree = degree
        self.shape_model = KernelRidge(alpha=self.alpha, kernel=self.kernel, degree=self.degree)
        self.measures_model = KernelRidge(alpha=self.alpha, kernel=self.kernel, degree=self.degree)
        self.shape_dirs = shape_dirs
        self.template = template
        self.faces = faces

    def per_vertex_shape_loss(self, actual_file, target_file):
        actual_mesh = trimesh.load(actual_file)
        target_mesh = trimesh.load(target_file)
        actual_vertices = actual_mesh.vertices
        target_vertices = target_mesh.vertices

        pervertex = []
        # for each_actual, each_target in zip(actual_vertices, target_vertices):
        #    pervertex.append(np.sum(np.abs(each_actual-each_target))/len(each_actual.flatten()))

        pervertex.append(np.sum(np.abs(actual_vertices - target_vertices)) / len(actual_vertices.flatten()))

        mae = np.mean(np.array(pervertex))
        std = np.std(abs(actual_vertices - target_vertices))
        # mape = mean_absolute_percentage_error(actual_vertices.flatten(), target_vertices.flatten())

        print(f"3D shape per vertex Error: \n Mean Absolute Error +/- std : {mae} +/- {std}")

        return mae, std


lock = multiprocessing.Lock()


def launch_test(args):
    folder1 = args[0]
    folder2 = args[1]
    indice = args[2]
    print("process", indice)
    print("folder1", folder1)
    print("folder2", folder2)

    height_tot = 0
    weight_tot = 0
    chest_tot = 0
    waist_tot = 0
    hip_tot = 0
    counter = 0

    height_list=[]
    weight_list = []
    chest_list = []
    waist_list = []
    hip_list = []

    for filename in os.listdir(folder1):  # C:\Users\lab2O\Documents\Francesco Manigrasso\polito\Tesi moro\3DAvatarGenerator sono uguali quindi devi mettere da 5000
        if filename.endswith('.obj'):

            # Check if corresponding mesh file exists in folder2
            # check dataset
            dataset = folder2.split("_")[8]
            index  = int(filename.split("_")[2].replace(".obj",""))
            index = index.__str__()

            filepath1 = os.path.join(folder1,filename )
            filepath2 = os.path.join(folder2,  "mesh_"+index+".obj")
            # filepath1="D:/Francesco Manigrasso/avatargenerator/Dataset/data10/human_body_meshes/male/subject_mesh_5001.obj"
            # filepath2 = "D:/Francesco Manigrasso/avatargenerator/Dataset/objects/avatargenerator_Dataset_data10_train_test_data_fold1001_male_data10_train_test_data_fold1001/mesh_0.obj"
            if os.path.exists(filepath2):
                # Load mesh files

                mesh1 = trimesh.load(filepath1)
                mesh2 = trimesh.load(filepath2)

                try:
                    height, weight, chest, waist, hip = calculate_measures(mesh1, mesh2)
                    height_tot = height_tot + height
                    weight_tot = weight_tot + weight
                    chest_tot = chest_tot + chest
                    waist_tot = waist_tot + waist
                    hip_tot = hip_tot + hip
                    counter = counter + 1
                    height_list.append(height)
                    weight_list.append(weight)
                    chest_list.append(chest)
                    waist_list.append(waist)
                    hip_list.append(hip)


                    """
                    print(f"height difference between {filename} is {height}")
                    print(f"weight difference between {filename} is {weight}")
                    print(f"chest difference between {filename} is {chest}")
                    print(f"waist difference between {filename} is {waist}")
                    print(f"hip difference between {filename} is {hip}")
                    """
                except:
                    print('exception')

    try:

        with lock:
            with open('obj_test_DB_std.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=';',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                print("folder1", folder1)
                print("folder2", folder2)

                print(f"cpunter value is: {counter}")
                height_tot = height_tot / counter
                weight_tot = weight_tot / counter
                chest_tot = chest_tot / counter
                waist_tot = waist_tot / counter
                hip_tot = hip_tot / counter
                print(f"mean height error: {height_tot}")
                print(f"mean weight error: {weight_tot}")
                print(f"mean chest error: {chest_tot}")
                print(f"mean waist error: {waist_tot}")
                print(f"mean hip error: {hip_tot}")

                row = [folder1, folder2, height_tot, np.std(height_list),weight_tot, np.std(weight_list) ,
                       chest_tot,np.std(chest_list), waist_tot,np.std(waist_list), hip_tot,np.std(hip_list), counter]

                writer.writerow(row)
                csvfile.flush()




    except:
        print("error in ", folder1, folder2)
        return Exception
    return 0


def main():
    # Leggi il dizionario di height e weight per ciascun soggetto femminile dal file JSON

    PATH = "D:/Francesco Manigrasso/avatargenerator/ScanDB_test_set/objects"
    PATH2 = "D:/Francesco Manigrasso/avatargenerator/ScanDB_test_set/meshes"
    result = [os.path.join(dp, f) for dp, dn, filenames in os.walk(PATH) for f in filenames if
              os.path.splitext(f)[1] == '.obj']

    completed_with_error = []

    folder_list = os.listdir(PATH)
    # open the file in the write mode

    iteration = 0
    readed = []

    # create the csv writer
    with open('obj_test_DB_std.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)

        if "folder2" not in readed:
            row = ["folder1", "folder2", "height_tot","height_std" ,"weight_tot","weight_std", "chest_tot",
                   "chest_std", "waist_tot","waist_std",
                   "hip_tot", "waist_std","counter"]

            writer.writerow(row)

            csvfile.flush()

        folders = []

        for f in folder_list:
            # print(folder_list.index(f), "on", len(folder_list))
            sets = f.split("_")
            folder1 = PATH2 + "/" + sets[7]
            # folder2 = './calvis_data/demo/test'
            # folder2 = './calvis_data/demo/original_test'
            folder2 = os.path.join(PATH, f)

            # folder1="D:/Francesco Manigrasso/avatargenerator/Dataset/data10/human_body_meshes/female"
            # folder2="D:/Francesco Manigrasso/avatargenerator/Dataset/objects/avatargenerator_Dataset_data10_train_test_data_fold3001_female_data10_train_test_data_fold1001"
            folders.append((folder1, folder2))
        """
        folders=[folders[9],folders[5],folders[3],folders[32],folders[34],folders[36],folders[38],folders[40],
                 folders[14],folders[12],folders[16],folders[18],folders[20]]
        """
        args = (b for b in folders)
        with tqdm(desc="Processing", total=len(folders)) as pbar:
            with ProcessPoolExecutor(max_workers=1) as executor:
                # results = list(executor.map(launch_test, folders))
                # results = executor.map(launch_test, folders)

                results = [
                    executor.submit(launch_test, args=(folders[i][0], folders[i][1], i))
                    for i in range(len(folders))
                ]
            k = 0
            for result in concurrent.futures.as_completed(results):
                print(f"{k}\t{result.result()}")
                k += 1
                pbar.update(1)


if __name__ == '__main__':
    main()
