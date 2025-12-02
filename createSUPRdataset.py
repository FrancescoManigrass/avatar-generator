#script con controllo doppioni

import os
import argparse
import numpy as np
from utils.model import *
from utils.torchloader import *
from array import *
import trimesh
import json

class Human():
    def __init__(self, shape_dirs, template, faces):
        self.shape_dirs = shape_dirs
        self.template = template
        self.faces = faces


    def create_3D(self, X):
        predicted_vertices = self.template + np.dot(self.shape_dirs, np.squeeze(X))
        return trimesh.Trimesh(predicted_vertices, self.faces)



def args():
    
    parser = argparse.ArgumentParser()
                        
    parser.add_argument('--gender', type = str, required = True,\
                        help='male or female')
    
    parser.add_argument('--num_shape', type = str, required = True,\
                        help='300 or 16 or 10')
    
    parser.add_argument('--num', type = str, required = True,\
                        help='number of subjects in the dataset')
    
    arguments = parser.parse_args()
    return arguments



def main():
    arguments = args()
    gender = arguments.gender
    num_shape = arguments.num_shape
    num = arguments.num

    template = np.load(f'data256/{gender}_template.npy')
    faces =  np.load(f'data256/faces.npy')
    shape_dirs = np.load(f'data256/{gender}_shapedirs.npy')
    
    shape = []
    tot_shape = []
    flag = 0
    j=0


    #aggiunta tuple betas del vecchio test set
    # data = json.load(open('data/train_test_data.json', 'r'))
    # for subject in data[gender]["test"]:
    #     measures_path = f"data/{gender}/{subject}/measures.json"
    #     m = json.load(open(measures_path, "r"))
    #     tot_shape.append(m['betas'])



    while j < int(num):
        flag=0
        shape = []
        #10-16-300 parametri per persona
        for i in range(int(num_shape)):
            rand=np.random.normal(0.0, 2.0)  #2   #4
            rand=np.clip(rand, -4, 4)  #-3 3   #-4 4
            rand=round(rand, 3)
            shape.append(rand)
        
        #controllo doppioni
        for t in range(j):
            if shape == tot_shape[t]:
                print('double')
                flag=1
                break
                #se c'è il doppione metto il flag a 1
        
        if int(flag) == 1:
            print('double')
            continue
        else:
            #se tutto va bene aggiungi la tupla e salva
            tot_shape.append(shape)
            #print(np.array(tot_shape).shape)

            shape1 = [shape]
            shape1 = np.array(shape1)

            human = Human(shape_dirs, template, faces)

            mesh = human.create_3D(shape1)

            #save shape values in a json file
            json_dict = {"betas": shape}
            
            numa=j
            numa=numa+1    
            numa=str(numa)
            numa=numa.zfill(4)

            with open(f"./dataset256/annotations/{gender}/subject_mesh_{numa}_anno.json", "w") as outfile:
                json.dump(json_dict, outfile)

            #mesh = mesh.smoothed()
            mesh.export(f'./dataset256/human_body_meshes/{gender}/subject_mesh_{numa}.obj')
            print("3D model saved!")

            j=j+1





if __name__ == "__main__":
    main()