import pickle
import argparse
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument("--pickle", type=str, required= True, help="path to pickle file")
    parser.add_argument("--gender", type=str, required= True, help="male or female")
    args = parser.parse_args()
    
    #pkl = pickle.load(open(args.pickle, 'rb'), encoding='latin1')
    
    try:
        os.mkdir('data')
    except:
        pass

    # #code for SMPL model
    # np.save(os.path.join('data', 'faces.npy'), pkl['f'])
    # np.save(os.path.join('data', f'{args.gender}_template.npy'), pkl['v_template'])
    # np.save(os.path.join('data', f'{args.gender}_shapedirs.npy'), pkl['shapedirs'])

    #code for SUPR model
    supr_path = f"SMPL/supr_{args.gender}.npy"
    nump = np.load(supr_path, allow_pickle=True)
    np.save(os.path.join('data256', 'faces.npy'), nump.item(0)['f'])
    shape_dirs = nump.item(0)['shapedirs']
    #np.save(os.path.join('data16', f'{args.gender}_shapedirs.npy'), shape_dirs[:,:,0:300])
    np.save(os.path.join('data256', f'{args.gender}_shapedirs.npy'), shape_dirs[:,:,0:256])
    np.save(os.path.join('data256', f'{args.gender}_template.npy'), nump.item(0)['v_template'])
    

