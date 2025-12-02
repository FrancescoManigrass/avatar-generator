import numpy as np
import json
from PIL import Image 
import trimesh
from shutil import copyfile
import os
import pickle
import argparse

def args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--resolution', type = int, required = True, default='512',\
                        help='for 300x300 image enter 300')
                        
    parser.add_argument('--gender', type = str, required = True,\
                        help='male or female')

    parser.add_argument('--loader_type', type = str, required = True,\
                        help='train, validation, test')
    
    parser.add_argument('--fold', type = str, required = True,\
                        help='name of the fold')
    
    parser.add_argument('--data_path', type = str, required = True,\
                        help='output folder')

    #new argument to choose wich type of weight to use
    #parser.add_argument('--weight', type = str, required = True, default='density',\
    #                    help='density or surface or volume')
    
    arguments = parser.parse_args()
    return arguments

def import_data(gender, loader_type, fold, data_path):
    """
    copies data from calvis folder to the data folder created.

    create a train_test seperated json file in the following format
    {
        male:{
            train:[sub_id1, sub_id2, ...],-
            test:[sub_id1, sub_id2, ...]
        },

        female:{
            train:[sub_id1, sub_id2, ...],
            test:[sub_id1, sub_id2, ...]
        }

    }.

    h_w_measures : run utils/measures_.py --path /path/to/the/obj/files --gender male/female.
    """
    print('entered import data')
    
    data = json.load(open(f'{data_path}/{fold}.json', 'r'))
    for subject in data[gender][loader_type]:
        print('importing')
        obj_file = f"dataset256/human_body_meshes/{gender}/subject_mesh_{subject}.obj"
        vertices = trimesh.load(obj_file).vertices
        print (f"{data_path}/{gender}/{subject}/vertices.npy")
        

        np.save(f"{data_path}/{gender}/{subject}/vertices.npy", vertices)

        measures = f"dataset256/annotations/{gender}/subject_mesh_{subject}_anno.json"
        destination = f"{data_path}/{gender}/{subject}/measures.json"

        copyfile(measures, destination)

def save_dataloaders(gender, resolution, loader_type, fold, data_path):
    """
    creates and save dataloaders (train/test) 
    """
    print('entered save data loaders')
    data = json.load(open(f'{data_path}/{fold}.json', 'r'))
    h_w_m = json.load(open(f"h_w_measures_{gender}_density_256.json", 'r'))
    front = []
    side = []
    h_w_measures = []
    
    ###################questo serviva a predire le dimensioni
    #measures = []
    
    vertices = []
    betas = []
    for subject in data[gender][loader_type]:
        print('saving')
        front_path = f"{data_path}/{gender}/{subject}/{resolution}/front.png"
        side_path = f"{data_path}/{gender}/{subject}/{resolution}/side.png"

        measures_path = f"{data_path}/{gender}/{subject}/measures.json"
        vertices_path = f"{data_path}/{gender}/{subject}/vertices.npy"
        
        front.append(np.array(Image.open(front_path).convert("L")))
        side.append(np.array(Image.open(side_path).convert("L")))

        h_w_measures.append(np.array(h_w_m[gender][f"subject_mesh_{subject}.obj"]))

        m = json.load(open(measures_path, "r"))

        ############ questi servivano a predire le dimensioni
        #measures.append(np.array([m["human_dimensions"]["chest_circumference"],\
        #m["human_dimensions"]["pelvis_circumference"], m["human_dimensions"]["waist_circumference"]]))

        betas.append(np.array(m['betas']))

        vertices.append(np.load(vertices_path))

    front = np.expand_dims(np.array(front), axis = -1)
    side = np.expand_dims(np.array(side), axis = -1)

    images = np.concatenate([front, side], axis = -1)
    
    #############questo serviva a predire le dimensioni
    #measures = np.array(measures)

    vertices = np.array(vertices, dtype="object")
    h_w_measures = np.array(h_w_measures)
    betas = np.array(betas)

    np.save(f"{data_path}/{fold}/dataloaders/{gender}/{loader_type}_{resolution}_images.npy", images)
    np.save(f"{data_path}/{fold}/dataloaders/{gender}/{loader_type}_vertices.npy", vertices)
    np.save(f"{data_path}/{fold}/dataloaders/{gender}/{loader_type}_h_w_measures_{gender}_density.npy", h_w_measures)
    
    ###########questo serviva a predire le dimensioni
    #np.save(f"CALVIS_data/dataloaders/{gender}/{loader_type}_measures.npy", measures)

    np.save(f"{data_path}/{fold}/dataloaders/{gender}/{loader_type}_betas.npy", betas)

def get_smpl_data(gender, data_path):
    """
    loads smpl template and save principal shapes, and template 
    """

    #code for SUPR model
    supr_path = f"smpl/supr_{gender}.npy"
    nump = np.load(supr_path, allow_pickle=True)
    shape_dirs = nump.item(0)['shapedirs']
    np.save(f"{data_path}/{gender}_shapedirs.npy", shape_dirs[:,:,0:256])
    np.save(f"{data_path}/faces.npy", nump.item(0)['f'])
    np.save(f"{data_path}/{gender}_template.npy", nump.item(0)['v_template'])

def main():
    arguments = args()
    gender = arguments.gender
    resolution = arguments.resolution
    loader_type = arguments.loader_type
    fold = arguments.fold 
    data_path = arguments.data_path
    #weight = arguments.weight
    try:
        os.mkdir(f"{data_path}/{fold}")
    except:
        pass

    try:
        os.mkdir(f"{data_path}/{fold}/dataloaders")
    except:
        pass

    try:
        os.mkdir(f"{data_path}/{fold}/dataloaders/{gender}")
    except:
        pass

    import_data(gender, loader_type, fold, data_path)
    get_smpl_data(gender, data_path)
    save_dataloaders(gender, resolution, loader_type, fold, data_path)
    print('FINISHED')
    
if __name__ == "__main__":
    main()

