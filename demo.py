from PIL import Image
import matplotlib.pyplot as plt
import os
import argparse
import torch
from measurement_evaluator import Human
import numpy as np
from joblib import load
from sklearn.decomposition import PCA
from utils.image_utils import ImgSizer
from utils.model import *
from utils.torchloader import *
import matplotlib.pyplot as plt
import trimesh


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--experiment", type=str, required=True, help='Give experiment name')
    parser.add_argument("--front_img", type=str, required=True, help="path to front view image")
    parser.add_argument("--side_img", type=str, required=True, help="path to side view image")
    parser.add_argument("--gender", type=str, required=True)
    parser.add_argument("--height", type=float, required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--feature_model", type=str, default='ae')
    parser.add_argument("--mesh_name", type=str, default='subject.obj')
    parser.add_argument("--measurement_model", type=str, default='calvis')
    parser.add_argument("--parameters", type=str, default='10')
    parser.add_argument("--split", type=str, default='train_test_data_fold1')

    args = parser.parse_args()

    lr = '0.0001'

    try:
        os.mkdir(os.path.join('calvis_data', 'demo'))
    except:
        pass

    try:
        os.mkdir(os.path.join('calvis_data/demo', args.experiment))
    except:
        pass

    front = np.array(Image.open(args.front_img).convert('L'),dtype='float16') / 255.0
    side = np.array(Image.open(args.side_img).convert('L'),dtype='float16') / 255.0


    front[front < 1] = 0
    front = 1 - front

    side[side < 1] = 0
    side = 1 - side

    print("Data preprocessed! \n Extracting Important features")

    if args.feature_model == 'ae':
        feature_extractor_path = (f'D:\\Francesco Manigrasso\\avatargenerator\\Dataset\\data{args.parameters}'
                                  f'\\{args.split}\\weights2\\0.0001\\new_base_feature_extractor_{args.gender}_200.pth')
        feature_extractor_weights = torch.load(feature_extractor_path, map_location=torch.device('cpu'))
        feature_extractor = Deep2DEncoder(image_size=512, kernel_size=3, n_filters=32)
        feature_extractor.load_state_dict(feature_extractor_weights)

        feature_extractor.eval()
        feature_extractor.requires_grad_(False)

        front = torch.tensor(front).float()
        side = torch.tensor(side).float()

        features = feature_extractor(front.view(1, 1, 512, 512))
        front_features = features.detach().numpy().reshape(256)

        # decoder_path = f'weights/decoder_female_50.pth'
        # decoder_weights = torch.load(decoder_path,  map_location=torch.device("cpu"))
        # decoder = Deep2DDecoder(image_size= 512 , kernel_size=3, n_filters=32)
        # decoder.load_state_dict(decoder_weights)

        # ###########
        # x_hat_front = decoder(features)
        # prova=x_hat_front[0,0,:,:].cpu().detach()
        # plt.imshow(prova)
        # plt.savefig(f'out_trainer/train/trial_front.png')

        features = feature_extractor(side.view(1, 1, 512, 512))
        side_features = features.detach().numpy().reshape(256)
        # ############
        # x_hat_side = decoder(features)
        # prova2=x_hat_side[0,0,:,:].cpu().detach()
        # plt.imshow(prova2)
        # plt.savefig(f'out_trainer/train/trial_side.png')

        front_features = np.array(front_features)
        side_features = np.array(side_features)

        features = np.concatenate([front_features, side_features], axis=-1).reshape(1, 512)

    if args.feature_model == 'pca':
        pca_front = load(f'weights/pca_{args.gender}_front.joblib')
        pca_side = load(f'weights/pca_{args.gender}_side.joblib')

        front_features = pca_front.transform(np.array(front).reshape(1, 512 * 512))
        side_features = pca_side.transform(np.array(side).reshape(1, 512 * 512))

        front_features = np.array(front_features)
        side_features = np.array(side_features)

        features = np.concatenate([front_features, side_features], axis=-1)
    print("Feature Extraction done \n Estimating Measurements")
    #template = np.load(f'calvis_data/{args.gender}_template.npy')
    #shape_dirs = np.load(f'calvis_data/{args.gender}_shapedirs.npy')
    #faces = np.load(f'calvis_data/faces.npy')

    if args.measurement_model == 'nomo':
        features = np.concatenate([features, np.array(args.height).reshape(1, 1)], axis=-1) #feautures dell'immagine del dataset 16
    else:
        features = np.concatenate([features, np.array(args.height).reshape(1, 1), np.array(args.weight).reshape(1, 1)],
                                  axis=-1)

    if args.measurement_model == 'nomo':
        human = load(f'weights/nomo_{args.gender}_krr.pkl') #modello utilizzato quindi 10

    else:

        """
        human = load(
            f'D:\\Francesco Manigrasso\\avatargenerator\\Dataset\\data{args.parameters}\\{args.split}\\features2\\{args.gender}\\new_base_{args.gender}_krr.pkl')
        """
        X_train = np.load( f'D:\\Francesco Manigrasso\\avatargenerator\\Dataset\\data{args.parameters}\\{args.split}\\features2\\{args.gender}\\ae_train_features.npy')
        X_train = X_train.squeeze()


        X_train_h_w = np.load(
            f'D:\\Francesco Manigrasso\\avatargenerator\\Dataset\\data{args.parameters}\\{args.split}\\dataloaders\\{args.gender}\\train_h_w_measures_{args.gender}_density.npy',
            allow_pickle=True)

        X_train = np.concatenate([X_train, X_train_h_w], axis=-1)  # X_train_h_w
        X_train = np.nan_to_num(X_train, copy=False)
        y_shape_train = np.load(f'D:\\Francesco Manigrasso\\avatargenerator\\Dataset\\data{args.parameters}\\{args.split}\\dataloaders\\{args.gender}\\train_betas.npy')


        template = np.load( f'D:\\Francesco Manigrasso\\avatargenerator\\Dataset\\data{args.parameters}\\{args.gender}_template.npy')
        shape_dirs = np.load(f'D:\\Francesco Manigrasso\\avatargenerator\\Dataset\\data{args.parameters}\\{args.gender}_shapedirs.npy')

        faces = np.load(f'D:\\Francesco Manigrasso\\avatargenerator\\Dataset\\data{args.parameters}\\faces.npy')

        human = Human(kernel='polynomial', \
                      alpha=1, \
                      degree=3, \
                      template=template, \
                      shape_dirs=shape_dirs, \
                      faces=faces)

        human.fit_shape(X_train, y_shape_train)



    #measurements = human.predict_measurements(features)
    shape = human.predict_shape(features)
    # print('*********shape**********')
    # print(shape)
    # print(type(shape))

    #print(f"Chest Circumference : {measurements[0][0]}")
    #print(f"Hip Circumference : {measurements[0][1]}")
    #print(f"Waist Circumference : {measurements[0][2]}")

    mesh = human.display_3D(shape)
    os.makedirs(args.experiment,exist_ok=True)
    mesh.export(os.path.join(f'{args.experiment}', args.mesh_name))
    print("3D model saved!")


if __name__ == "__main__":
    main()


















