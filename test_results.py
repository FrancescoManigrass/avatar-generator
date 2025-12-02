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

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error
import numpy as np
import trimesh
import argparse
from joblib import dump, load
import pandas as pd

from body_measurements.measurement import Body3D
import numpy as np
import trimesh
import subprocess
import os
import json

from body_measurements.measurement import Body3D

def calculate_measures(mesh1, mesh2):
    o_body = Body3D(mesh1.vertices, mesh1.faces)
    r_body = Body3D(mesh2.vertices, mesh2.faces)

    o_height = o_body.height()
    r_height = r_body.height()
    height = np.abs(o_height-r_height)

    o_weight = o_body.weight()
    r_weight = r_body.weight()
    weight = np.abs(o_weight-r_weight)

    _, chest_location,o_chest = o_body.chest()
    _, chest_location,r_chest = r_body.chest()
    chest = np.abs(o_chest-r_chest)

    _, waist_location,o_waist = o_body.waist()
    _, waist_location,r_waist = r_body.waist()
    waist = np.abs(o_waist-r_waist)

    _, hip_location,o_hip = o_body.hip()
    _, hip_location,r_hip = r_body.hip()
    hip = np.abs(o_hip-r_hip)

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
        self.shape_model = KernelRidge(alpha = self.alpha, kernel = self.kernel, degree = self.degree)
        self.measures_model = KernelRidge(alpha = self.alpha, kernel = self.kernel, degree = self.degree)
        self.shape_dirs = shape_dirs
        self.template = template
        self.faces = faces

    def per_vertex_shape_loss(self, actual_file, target_file):
        actual_mesh = trimesh.load(actual_file)
        target_mesh = trimesh.load(target_file)
        actual_vertices = actual_mesh.vertices
        target_vertices = target_mesh.vertices

        pervertex = []
        #for each_actual, each_target in zip(actual_vertices, target_vertices):
        #    pervertex.append(np.sum(np.abs(each_actual-each_target))/len(each_actual.flatten()))

        pervertex.append(np.sum(np.abs(actual_vertices-target_vertices))/len(actual_vertices.flatten()))

        mae = np.mean(np.array(pervertex))
        std = np.std(abs(actual_vertices - target_vertices))
        #mape = mean_absolute_percentage_error(actual_vertices.flatten(), target_vertices.flatten())

        print(f"3D shape per vertex Error: \n Mean Absolute Error +/- std : {mae} +/- {std}")

        return mae, std




def main():
    # Leggi il dizionario di height e weight per ciascun soggetto femminile dal file JSON
    with open('calvis_h_w_measures_female_density.json') as f:
        data = json.load(f)["female"]

    # Eseguire la demo per ciascun soggetto femminile del test set
    for filename in data:
        subject_index = filename.split('_')[2].split('.')[0]
        #subject_index = filename.split('.')[0]
        if int(subject_index) >= 1863 and int(subject_index) <= 5000:
        #if int(subject_index) >= 0 and int(subject_index) <= 9:
            height, weight = data[filename]

            #le immagini front e side si trovano in input_folder/cartella con nome del soggetto
            #ciascuna immagine si chiama front e side
            input_folder = f"./input/mine_oldR/{subject_index}/512/"
            front = input_folder + "front.png"
            side = input_folder + "side.png"

            #il nome della la mesh in output sarà il seguente
            mesh_name = f"subject_mesh_{subject_index}.obj"

            command = f"python demo.py --gender female --height {height} --weight {weight} --front {front} --side {side} --mesh_name {mesh_name} --experiment mine_oldR"
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()



    folder1 = './female_test_set'
    #folder2 = './calvis_data/demo/test'
    #folder2 = './calvis_data/demo/original_test'
    folder2 = './calvis_data/demo/mine_oldR'

    mean_distance = 0

    height1=0
    height2=0
    sum_height=0

    weight1=0
    weight2=0
    sum_weight=0

    chest1=0
    chest2=0
    sum_chest=0

    waist1=0
    waist2=0
    sum_waist=0

    hip1=0
    hip2=0
    sum_hip=0

    mae=0
    std=0
    mae_tot=0
    std_tot=0

    height_tot=0
    weight_tot=0
    chest_tot=0
    waist_tot=0
    hip_tot=0
    counter=0


    for filename in os.listdir(folder1):
        if filename.endswith('.obj'):
            # Check if corresponding mesh file exists in folder2
            filepath1 = os.path.join(folder1, filename)
            filepath2 = os.path.join(folder2, filename)
            if os.path.exists(filepath2):
                # Load mesh files
                print("entrato")
                mesh1 = trimesh.load(filepath1)
                mesh2 = trimesh.load(filepath2)

                try:
                  height, weight, chest, waist, hip=calculate_measures(mesh1, mesh2)
                  height_tot=height_tot+height
                  weight_tot=weight_tot+weight
                  chest_tot=chest_tot+chest
                  waist_tot=waist_tot+waist
                  hip_tot=hip_tot+hip
                  counter=counter+1

                  print(f"height difference between {filename} is {height}")
                  print(f"weight difference between {filename} is {weight}")
                  print(f"chest difference between {filename} is {chest}")
                  print(f"waist difference between {filename} is {waist}")
                  print(f"hip difference between {filename} is {hip}")
                except:
                  print('exception')



                """
                template = np.load(f'calvis_data/female_template.npy')
                shape_dirs = np.load(f'calvis_data/female_shapedirs.npy')
                faces =  np.load(f'calvis_data/faces.npy')
                human = Human(kernel = 'polynomial',\
                  alpha = 1.0,\
                  degree = 3,\
                  template = template,\
                  shape_dirs = shape_dirs,\
                  faces = faces)
                
                human = load(f'weights/original/calvis_female_krr.pkl')

                mae, std = human.per_vertex_shape_loss(filepath1, filepath2)
                mae_tot=mae_tot+mae
                std_tot=std_tot+std

                

                # Calculate Hausdorff distance
                distance = hausdorff_distance(mesh1, mesh2)
                #aggiorna somma distanze
                mean_distance = mean_distance + distance
                # Print result
                print(f"Hausdorff distance between {filename} is {distance}")
                

                
                #crea istanza di body
                body1 = Body3D(mesh1.vertices, mesh1.faces)
                body2 = Body3D(mesh2.vertices, mesh2.faces)

                #calcola altezza di mesh1
                #calcola altezza di mesh2
                #fai la differenza
                #salva in un array
                height1=body1.height()
                height2=body2.height()
                difference_height=abs(height1-height2)
                sum_height=sum_height+difference_height             

                #ripeti per le altre misure
                weight1=body1.weight()
                weight2=body2.weight()
                difference_weight=abs(height1-height2)
                sum_weight=0

                chest1=body2.chest()
                chest2=body2.chest()
                difference_chest=abs(chest1-chest2)
                sum_chest=sum_chest+difference_chest

                waist1=body1.waist()
                waist2=body2.waist()
                difference_waist=abs(waist1-waist2)
                sum_waist=sum_waist+difference_waist

                hip1=body1.hip()
                hip2=body2.hip()
                difference_hip=abs(hip1-hip2)
                sum_hip=sum_hip+difference_hip
                """
    """
    mae_tot = mae_tot/421
    std_tot = std_tot/421
    print(f"3D shape per vertex Error TOTAL: \n Mean Absolute Error +/- std : {np.round(mae_tot, 5)} +/- {np.round(std_tot, 5)}")
    
    #calcola valore medio delle differenze
    sum_height=sum_height/421
    sum_weight=sum_weight/421
    sum_chest=sum_chest/421
    sum_waist=sum_waist/421
    sum_hip=sum_hip/421
    print(f"height error: {sum_height}")
    print(f"weight error: {sum_weight}")
    print(f"chest error: {sum_chest}")
    print(f"waist error: {sum_waist}")
    print(f"hip error: {sum_hip}")
    

    
    # calcola valore medio
    mean_distance = mean_distance/1000
    # Print result
    print(f"Mean Hausdorff distance among the test files is {mean_distance}")
    """

    print(f"cpunter value is: {counter}")
    height_tot=height_tot/counter
    weight_tot=weight_tot/counter
    chest_tot=chest_tot/counter
    waist_tot=waist_tot/counter
    hip_tot=hip_tot/counter
    print(f"mean height error: {height_tot}")
    print(f"mean weight error: {weight_tot}")
    print(f"mean chest error: {chest_tot}")
    print(f"mean waist error: {waist_tot}")
    print(f"mean hip error: {hip_tot}")



if __name__ == '__main__':
	main()