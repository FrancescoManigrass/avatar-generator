from body_measurements.measurement import Body3D
import trimesh
import json
import os
import argparse

def calculate_measures(mesh):
    o_body = Body3D(mesh.vertices, mesh.faces)
    
    o_height = o_body.height()
    
    o_weight = o_body.weight()
    
    _, chest_location,o_chest = o_body.chest()
    
    _, waist_location,o_waist = o_body.waist()
    
    _, hip_location,o_hip = o_body.hip()
    
    
    return o_height, o_weight, o_chest, o_waist, o_hip




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--gender', type = str, required = True,help='male or female')
    parser.add_argument('--path', type = str, required = True, help='path to original obj files')
    args = parser.parse_args()

                        
    gender_measures = dict()
    measures = dict()
    chest = 0
    waist = 0
    hip = 0
    weight = 0
    height = 0



    for subject in sorted(os.listdir(args.path)):
        filename = os.path.join(args.path, subject)
        print('filename:', filename)
        mesh = trimesh.load(filename)
        try:
            # body = Body3D(mesh.vertices, mesh.faces)

            # height = body.height()
            # print(height)

            # weight = body.weight()
            # print(weight)

            # chest = body.chest()
            # print(chest)

            # waist = body.waist()
            # print(waist)

            # hip = body.hip()
            # print(hip)

            height, weight, chest, waist, hip=calculate_measures(mesh)
            print(f'height: {height}')
            print(f'weight: {weight}')
            print(f'chest: {chest}')
            print(f'waist: {waist}')
            print(f'hip: {hip}')


        except:
            print('exception')
            chest = None
            waist = None
            hip = None
            weight = None
            height = None

        measures[subject] = [height, weight, chest, waist, hip]
    gender_measures[args.gender] = measures


    with open(f'dataset4_measures_{args.gender}_plus.json', 'w') as fp:
        print('entrato nella stampa')
        json.dump(gender_measures, fp)
        print('stampato')
    


        
