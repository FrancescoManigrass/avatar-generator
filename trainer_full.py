import argparse
import os
from os.path import join
from joblib import dump, load
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils.torchloader import *
from utils.model import *
import cv2
import warnings
from time import sleep
import time
from utils.losses import MAASE, Accuracy
from sklearn.decomposition import PCA
from joblib import dump
import logging
import torch
import neptune.new as neptune


run = None

def cc(image):
    for i in range(16):
        image = image[i, 0, :, :]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        output = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S)
        (numLabels, labels, stats, centroids) = output
        mask = torch.ones(gray.shape, dtype="uint8")
        componentMask = (labels == 1).astype("uint8") * 255
        mask = torch.bitwise_or(mask, componentMask)
        mask = torch.bitwise_not(mask)
        return mask


warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


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

    def fit_measurements(self, X, y):
        # sum0 = sum(pd.isna(X))
        # print("Checking for NaN in X : ", sum0)
        where_are_NaNs = pd.isna(X)
        X[where_are_NaNs] = 0
        sum0 = sum(pd.isna(X))
        # print("Checking for NaN in X : ", sum0)
        return self.measures_model.fit(X, y)

    def fit_shape(self, X, y):
        where_are_NaNs = pd.isna(X)
        X[where_are_NaNs] = 0
        return self.shape_model.fit(X, y)

    def predict_measurements(self, X):
        return self.measures_model.predict(X)

    def predict_shape(self, X):
        return self.shape_model.predict(X)

    def display_3D(self, X):
        predicted_vertices = self.template + np.dot(self.shape_dirs, np.squeeze(X))
        # trimesh.Trimesh(predicted_vertices, self.faces).show()
        return trimesh.Trimesh(predicted_vertices, self.faces)

    def predict_3D_vertices(self, target, actual):

        p_vertices = []
        a_vertices = []
        for t, a in zip(target, actual):
            # print('dimensioni')
            # print(self.template.shape)
            # print(self.shape_dirs.shape)
            # print(target.shape)
            # print(t.shape)
            # print(np.squeeze(t).shape)

            predicted_vertices = self.template + np.dot(self.shape_dirs, np.squeeze(t))
            actual_vertices = self.template + np.dot(self.shape_dirs, np.squeeze(a))
            p_vertices.append(predicted_vertices)
            a_vertices.append(actual_vertices)
        return np.array(p_vertices), np.array(a_vertices)

    def measurement_loss(self, actual, target):
        mae = mean_absolute_error(actual, target)
        std = np.std(abs(actual - target))

        chest_error = mean_absolute_error(actual[:, 0], target[:, 0])
        hip_error = mean_absolute_error(actual[:, 1], target[:, 1])
        waist_error = mean_absolute_error(actual[:, 2], target[:, 2])

        chest_std = np.std(abs(actual[:, 0] - target[:, 0]))
        hip_std = np.std(abs(actual[:, 1] - target[:, 1]))
        waist_std = np.std(abs(actual[:, 2] - target[:, 2]))

        print(f"Measurements Error:")
        print()
        print(f"Over ALL MAE +/- std : {np.round(mae * 1000, 2)} +/- {np.round(std * 1000, 2)} mm")
        print(f"Chest MAE +/- std : {np.round(chest_error * 1000, 2)} +/- {np.round(chest_std * 1000, 2)} mm")
        print(f"Hip MAE +/- std : {np.round(hip_error * 1000, 2)} +/- {np.round(hip_std * 1000, 2)} mm")
        print(f"Waist MAE +/- std : {np.round(waist_error * 1000, 2)} +/- {np.round(waist_std * 1000, 2)} mm")

    def shape_parameters_loss(self, actual, target):
        logging.basicConfig(filename=f"log_measurement_evaluator_{args.gender}.txt", level=logging.INFO)
        mae = mean_absolute_error(actual, target)
        std = np.std(abs(actual - target))
        print()

        return mae,std

    def per_vertex_shape_loss(self, actual, target):
        logging.basicConfig(filename=f"log_measurement_evaluator_{args.gender}.txt", level=logging.INFO)
        pervertex = []
        for each in np.abs(actual - target):
            pervertex.append(np.sum(each) / len(each.flatten()))

        mae = np.mean(np.array(pervertex))
        std = np.std(abs(actual - target))
        mape = mean_absolute_percentage_error(actual.flatten(), target.flatten())

        return mae,std,mape


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss

        return Dice_BCE


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        DICE = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return DICE


class AETrainer(object):
    def __init__(self, device, gpu, gender, batch_size, loss, loss2):
        self.device = device
        self.gender = gender
        self.gpu = gpu
        self.batch_size = batch_size

        losses = {'mse': nn.MSELoss(), \
                  'mae': nn.L1Loss(), \
                  'mae+mse': MAASE(), \
                  'bce': nn.BCELoss(), \
                  'diceBCE': DiceBCELoss(), \
                  'dice': DiceLoss()}

        self.loss = losses[loss]
        self.loss2 = losses['bce']
        self.epochs = None
        self.accuracy = Accuracy()

    def train_model(self, data_loader, test_data_loader, args, epochs=1000, learning_rate=0.0001, betas=(0.9, 0.99),
                    folder=""):

        lr = str(learning_rate)

        run = neptune.init_run(
            project="GRAINS/3DAvatarGenerator",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZmI1MzQwMS0xZTdiLTRkZWQtOGQ4Yi0yNWE0ZGQ1MzEzNTQifQ==",



        )



        hyperparameters = {'n_epochs': epochs, 'learning_rate': learning_rate, 'model': 'AE', 'loss': 'bce',
                           'batch_size': args.batch_size.__str__(), 'seed': torch.seed().__str__(),
                           'dataset':args.path_folder.__str__(),
                           "gender": args.gender.__str__()}
        run['hyperparameters'] = hyperparameters
        run["config_files"].upload_files(
            [os.path.join(dp, f) for dp, dn, filenames in os.walk(".") for f in filenames if
             os.path.splitext(f)[1] == '.py'])

        run["train_file"].upload("trainer_full.py")

        # image_size is the size of input mask
        self.epochs = epochs     

        feature_extractor = Deep2DEncoder(image_size=512, kernel_size=3, n_filters=32)
        decoder = Deep2DDecoder(image_size=512, kernel_size=3, n_filters=32)

        feature_extractor.to(self.device)
        decoder.to(self.device)

        optimizer_extractor = torch.optim.Adam(feature_extractor.parameters(), lr=learning_rate, betas=betas)
        optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr=learning_rate, betas=betas)    
      
        reconstruction_criterion2 = self.loss2

        cuda = True if torch.cuda.is_available() else False

        if cuda:
            ##reconstruction_criterion1.cuda()
            reconstruction_criterion2.cuda()

        start_time = time.time()


        for epoch in range(epochs):
            feature_extractor.train()
            decoder.train()
            feature_extractor.requires_grad_(True)
            decoder.requires_grad_(True)

            step = 0
            total_accuracy_front = 0
            total_accuracy_side = 0
            total_recon_loss_dice = 0
            total_recon_loss_bce = 0
            total_recon_loss = 0

            for x_front, x_side in tqdm(data_loader):
                x_front = x_front.to(device=self.device)
                x_side = x_side.to(device=self.device)
                feature_front = feature_extractor(x_front)
                feature_side = feature_extractor(x_side)

                x_hat_front = decoder(feature_front)
                x_hat_side = decoder(feature_side)

                # print(x_hat_front.shape)
                # print(x_hat_front)

                # connected components su x_hat_front e x_hat_side
                # ottengo l'immagine nuova sulla quale applico la dice loss
                # mask_front = cc(x_hat_front)
                # mask_side = cc(x_hat_side)

                # commentare nel caso di utilizzo di modello preallenato
                ##recon_loss_front_dice = reconstruction_criterion1(x_hat_front, x_front)
                ##recon_loss_side_dice = reconstruction_criterion1(x_hat_side, x_side)
                recon_loss_front_bce = reconstruction_criterion2(x_hat_front, x_front)
                recon_loss_side_bce = reconstruction_criterion2(x_hat_side, x_side)

                # nuova recon_loss solo con la dice
                # recon_loss_front = reconstruction_criterion(mask_front, x_front)
                # recon_loss_side = reconstruction_criterion(mask_side, x_side)
                ##recon_loss_dice = recon_loss_front_dice+recon_loss_side_dice
                recon_loss_bce = recon_loss_front_bce + recon_loss_side_bce

                # recon_loss = recon_loss_dice+(recon_loss_bce*0.5)
                recon_loss = recon_loss_bce

                optimizer_extractor.zero_grad()
                optimizer_decoder.zero_grad()
                recon_loss.backward()
                optimizer_extractor.step()
                optimizer_decoder.step()

                total_accuracy_front += self.accuracy(x_hat_front, x_front)
                total_accuracy_side += self.accuracy(x_hat_side, x_side)
                ##total_recon_loss_dice += recon_loss_dice.item()
                total_recon_loss_bce += recon_loss_bce.item()
                total_recon_loss += recon_loss.item()

                step += 1

                # run['TRAIN accuracy front'].log(self.accuracy(x_hat_front, x_front))
                # run['TRAIN accuracy side'].log(self.accuracy(x_hat_side, x_side))
                ##run['TRAIN recon_loss_dice'].log(recon_loss_dice.item())
                # run['TRAIN recon_loss_bce'].log(recon_loss_bce.item())
                # run['TRAIN recon_loss'].log(recon_loss.item())

            # TESTING
            feature_extractor.eval()
            decoder.eval()
            feature_extractor.requires_grad_(False)
            decoder.requires_grad_(False)

            step_test = 0
            total_accuracy_front_test = 0
            total_accuracy_side_test = 0
            total_recon_loss_dice_test = 0
            total_recon_loss_bce_test = 0
            total_recon_loss_test = 0

            if epoch % 10 == 0 :

                for x_front, x_side in tqdm(test_data_loader):
                    x_front = x_front.to(device=self.device)
                    x_side = x_side.to(device=self.device)
                    feature_front = feature_extractor(x_front)
                    feature_side = feature_extractor(x_side)

                    x_hat_front = decoder(feature_front)
                    x_hat_side = decoder(feature_side)

               
                    recon_loss_front_bce = reconstruction_criterion2(x_hat_front, x_front)
                    recon_loss_side_bce = reconstruction_criterion2(x_hat_side, x_side)

               
                    recon_loss_bce = recon_loss_front_bce + recon_loss_side_bce

                   
                    recon_loss = recon_loss_bce

                    total_accuracy_front_test += self.accuracy(x_hat_front, x_front)
                    total_accuracy_side_test += self.accuracy(x_hat_side, x_side)
                 
                    total_recon_loss_bce_test += recon_loss_bce.item()
                    total_recon_loss_test += recon_loss.item()

                    step_test += 1

                    # run['TEST accuracy front'].log(self.accuracy(x_hat_front, x_front))
                    # run['TEST accuracy side'].log(self.accuracy(x_hat_side, x_side))
                    ##run['TEST recon_loss_dice'].log(recon_loss_dice.item())
                    # run['TEST recon_loss_bce'].log(recon_loss_bce.item())
                    # run['TEST recon_loss'].log(recon_loss.item())

                #

                # print('Epoch {}/{} Time={:.2f}s train_loss_front={:.4f} train_loss_side={:.4f} accuracy_front= {:.4f} accuracy_side= {:.4f}'.format(
                #                         epoch +1, epochs,
                #                         elapsed_time,
                #                         epoch_loss_front, epoch_loss_side, total_accuracy_front, total_accuracy_side))

                run['Total TEST accuracy front'].log(total_accuracy_front_test / step_test)
                run['Total TEST accuracy side'].log(total_accuracy_side_test / step_test)
                ##run['Total TEST recon_loss_dice'].log(total_recon_loss_dice_test/step_test)
                ##run['Total TEST recon_loss_bce'].log(total_recon_loss_bce_test/step_test)
                run['Total TEST recon_loss'].log(total_recon_loss_test / step_test)

                os.makedirs(folder + "/" + f'weights3/{lr}', exist_ok=True)

                torch.save(feature_extractor.state_dict(),
                           folder + "/" + f'weights3/{lr}/new_base_feature_extractor_{self.gender}_{epochs}.pth')
                torch.save(decoder.state_dict(),
                           folder + "/" + f'weights3/{lr}/new_base_decoder_{self.gender}_{epochs}.pth')
                torch.save(optimizer_extractor.state_dict(),
                           folder + "/" + f'weights3/{lr}/new_base_extractor_opt_{self.gender}_{epochs}.pth')
                torch.save(optimizer_decoder.state_dict(),
                           folder + "/" + f'weights3/{lr}/new_base_decoder_opt_{self.gender}_{epochs}.pth')

                if args.model == 'ae':
                    model = AEXtractor(device=device, batch_size=args.batch_size, gender=args.gender, loss=args.loss,
                                       loss2='bce', path_folder=args.path_folder, lr=args.lr)

                    if args.mode == 'performance':

                        dataset = Data(args.data_path)
                        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                num_workers=8)
                        print("AE Evaluation Started")
                        model.test_performance(dataloader)
                        print("Evaluation Done")

                    else:
                        def create_features(loader_type):
                            mode = args.data_path.split("/")[-1].split('_')[0]
                            data = np.load(
                                args.path_folder + "/dataloaders/" + args.gender + "/" + loader_type + "_512_images.npy")
                            data = np.array(data, dtype='float16') / 255.0
                            data[data < 1] = 0
                            data = 1 - data

                            print("Extracting Encoder features3")
                            features = model.extract_features(data)

                            os.makedirs(args.path_folder + "/features3" + "/" + args.gender, exist_ok=True)
                            np.save(
                                f"{args.path_folder}/features3/{args.gender}/{args.model}_{loader_type}_features.npy",
                                features)

                        create_features("train")
                        create_features("validation")
                        create_features("test")

                        print("Extraction Done")

                template = np.load("/".join(args.path_folder.split("/")[:-1])+'/' + args.gender + '_template.npy')
                shape_dirs = np.load("/".join(args.path_folder.split("/")[:-1])+'/' + args.gender +  '_shapedirs.npy')

                """
                template = np.load(
                    args.path_folder.split("/")[0] + "/" + args.path_folder.split("/")[
                        1] + '/' + args.gender + '_template.npy')
                shape_dirs = np.load(args.path_folder.split("/")[0] + "/" + args.path_folder.split("/")[
                    1] + '/' + args.gender + '_shapedirs.npy')
                faces = np.load(args.path_folder.split("/")[0] + "/" + args.path_folder.split("/")[1] + '/faces.npy')
                
                """

                faces = np.load("/".join(args.path_folder.split("/")[:-1])+ '/faces.npy')

                human = Human(kernel='polynomial', \
                              alpha=1, \
                              degree=3, \
                              template=template, \
                              shape_dirs=shape_dirs, \
                              faces=faces)

                X_train = np.load(f"{args.path_folder}/features3/{args.gender}/ae_train_features.npy")
                X_train = X_train.squeeze()
                X_test = np.load(f"{args.path_folder}/features3/{args.gender}/ae_test_features.npy")
                X_test = X_test.squeeze()

                X_validation = np.load(f"{args.path_folder}/features3/{args.gender}/ae_validation_features.npy")
                X_validation = X_validation.squeeze()

                X_train_h_w = np.load(
                    f'{args.path_folder}/dataloaders/{args.gender}/train_h_w_measures_{args.gender}_density.npy',
                    allow_pickle=True)
                X_test_h_w = np.load(
                    f'{args.path_folder}/dataloaders/{args.gender}/test_h_w_measures_{args.gender}_density.npy',
                    allow_pickle=True)
                X_validation_h_w = np.load(
                    f'{args.path_folder}/dataloaders/{args.gender}/validation_h_w_measures_{args.gender}_density.npy',
                    allow_pickle=True)

                X_train = np.concatenate([X_train, X_train_h_w], axis=-1)  # X_train_h_w
                X_test = np.concatenate([X_test, X_test_h_w], axis=-1)
                X_validation = np.concatenate([X_validation, X_validation_h_w], axis=-1)

                X_train = np.nan_to_num(X_train, copy=False)
                X_test = np.nan_to_num(X_test)
                X_validation = np.nan_to_num(X_validation)

                ########questi servivano per predire le misure
                # y_measures_train = np.load(f'data/dataloaders/{args.gender}/train_measures.npy')
                # y_measures_test = np.load(f'data/dataloaders/{args.gender}/test_measures.npy')

                y_shape_train = np.load(f'{args.path_folder}/dataloaders/{args.gender}/train_betas.npy')
                y_shape_test = np.load(f'{args.path_folder}/dataloaders/{args.gender}/test_betas.npy')
                y_shape_validation = np.load(f'{args.path_folder}/dataloaders/{args.gender}/validation_betas.npy')

                human.fit_shape(X_train, y_shape_train)

                ########questo serviva per predire le misure
                # dump(human, f'data/dataloaders/{args.gender}/data/10_bceNoNormae_{args.gender}_krr.pkl')
                dump(human, f'{args.path_folder}/features3/{args.gender}/new_base_{args.gender}_krr.pkl')

                del human

                human = load(f'{args.path_folder}/features3/{args.gender}/new_base_{args.gender}_krr.pkl')

                ########questo serviva per predire le misure
                # measurements = human.predict_measurements(X_train)
                shape = human.predict_shape(X_train)
                p_verts, a_verts = human.predict_3D_vertices(shape, y_shape_train)

                print("TRAIN")
                ########questo serviva per predire le misure
                # human.measurement_loss(measurements, y_measures_train)
                mae, std = human.shape_parameters_loss(shape, y_shape_train)
                run['TRAIN_Shape Per Parameters Error-mae'].log(mae)
                run['TRAIN_Shape Per Parameters Error-std'].log(std)
                print(
                    f"Shape Per Parameters Error: \n Mean Absolute Error +/- std : {np.round(mae, 5)} +/- {np.round(std, 5)}")
                mae, std, mape = human.per_vertex_shape_loss(p_verts, a_verts)
                print()
                print(
                    f"3D shape per vertex Error: \n Mean Absolute Error +/- std : {np.round(mae, 5)} +/- {np.round(std, 5)}")
                print(f"Mape : {np.round(mape)}")
                run['TRAIN_3D shape per vertex Error-mae'].log(mae)
                run['TRAIN_3D shape per vertex Error-std'].log(std)
                run['TRAIN_Mape'].log(mape)

                # aggiunta per la gestione dei missing values nelle misure se si vogliono aggiungere ulteriori dati per aiutare la predizioen
                # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
                # imp_mean.fit(X_train)
                # X_test = imp_mean.transform(X_test)

                ########questo serviva per predire le misure
                # measurements = human.predict_measurements(X_test)
                shape = human.predict_shape(X_validation)
                p_verts, a_verts = human.predict_3D_vertices(shape, y_shape_validation)

                print("VALIDATION")
                ########questo serviva per predire le misure
                # human.measurement_loss(measurements, y_measures_test)
                mae, std = human.shape_parameters_loss(shape, y_shape_validation)
                run['VALIDATION_Shape Per Parameters Error-mae'].log(mae)
                run['VALIDATION_Shape Per Parameters Error-std'].log(std)
                print(
                    f"Shape Per Parameters Error: \n Mean Absolute Error +/- std : {np.round(mae, 5)} +/- {np.round(std, 5)}")
                mae, std, mape = human.per_vertex_shape_loss(p_verts, a_verts)
                print()
                print(
                    f"3D shape per vertex Error: \n Mean Absolute Error +/- std : {np.round(mae, 5)} +/- {np.round(std, 5)}")
                print(f"Mape : {np.round(mape)}")
                run['VALIDATION_3D shape per vertex Error-mae'].log(mae)
                run['VALIDATION_3D shape per vertex Error-std'].log(std)
                run['VALIDATION_Mape'].log(mape)




            elapsed_time = time.time() - start_time
            print('Epoch {}/{} Time={:.2f}s'.format(epoch + 1, epochs, elapsed_time))

            run['Total TRAIN accuracy front'].log(total_accuracy_front / step)
            run['Total TRAIN accuracy side'].log(total_accuracy_front / step)
            ##run['Total TRAIN recon_loss_dice'].log(total_recon_loss_dice/step)
            ##run['Total TRAIN recon_loss_bce'].log(total_recon_loss_bce/step)
            run['Total TRAIN recon_loss'].log(total_recon_loss / step)

        

        



        

        human = Human(kernel='polynomial', \
                      alpha=1, \
                      degree=3, \
                      template=template, \
                      shape_dirs=shape_dirs, \
                      faces=faces)

        human.fit_shape(X_train, y_shape_train)

        ########questo serviva per predire le misure
        # dump(human, f'data/dataloaders/{args.gender}/data/10_bceNoNormae_{args.gender}_krr.pkl')
        dump(human, f'{args.path_folder}/features3/{args.gender}/new_base_{args.gender}_krr.pkl')

        del human

        human = load(f'{args.path_folder}/features3/{args.gender}/new_base_{args.gender}_krr.pkl')

        ########questo serviva per predire le misure
        # measurements = human.predict_measurements(X_train)
        shape = human.predict_shape(X_train)
        p_verts, a_verts = human.predict_3D_vertices(shape, y_shape_train)

        print("TRAIN")
        ########questo serviva per predire le misure
        # human.measurement_loss(measurements, y_measures_train)
        mae,std=human.shape_parameters_loss(shape, y_shape_train)
        run['TRAIN_Shape Per Parameters Error-mae'].log(mae)
        run['TRAIN_Shape Per Parameters Error-std'].log(std)
        print(f"Shape Per Parameters Error: \n Mean Absolute Error +/- std : {np.round(mae, 5)} +/- {np.round(std, 5)}")
        mae, std,mape = human.per_vertex_shape_loss(p_verts, a_verts)
        print()
        print(f"3D shape per vertex Error: \n Mean Absolute Error +/- std : {np.round(mae, 5)} +/- {np.round(std, 5)}")
        print(f"Mape : {np.round(mape)}")
        run['TRAIN_3D shape per vertex Error-mae'].log(mae)
        run['TRAIN_3D shape per vertex Error-std'].log(std)
        run['TRAIN_Mape'].log(mape)

        # aggiunta per la gestione dei missing values nelle misure se si vogliono aggiungere ulteriori dati per aiutare la predizioen
        # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
        # imp_mean.fit(X_train)
        # X_test = imp_mean.transform(X_test)

        ########questo serviva per predire le misure
        # measurements = human.predict_measurements(X_test)
        shape = human.predict_shape(X_validation)
        p_verts, a_verts = human.predict_3D_vertices(shape, y_shape_validation)

        print("VALIDATION")
        ########questo serviva per predire le misure
        # human.measurement_loss(measurements, y_measures_test)
        mae, std = human.shape_parameters_loss(shape, y_shape_validation)
        run['VALIDATION_Shape Per Parameters Error-mae'].log(mae)
        run['VALIDATION_Shape Per Parameters Error-std'].log(std)
        print(f"Shape Per Parameters Error: \n Mean Absolute Error +/- std : {np.round(mae, 5)} +/- {np.round(std, 5)}")
        mae, std, mape = human.per_vertex_shape_loss(p_verts, a_verts)
        print()
        print(f"3D shape per vertex Error: \n Mean Absolute Error +/- std : {np.round(mae, 5)} +/- {np.round(std, 5)}")
        print(f"Mape : {np.round(mape)}")
        run['VALIDATION_3D shape per vertex Error-mae'].log(mae)
        run['VALIDATION_3D shape per vertex Error-std'].log(std)
        run['VALIDATION_Mape'].log(mape)

        print("TEST")
        shape = human.predict_shape(X_test)
        p_verts, a_verts = human.predict_3D_vertices(shape, y_shape_test)
        ########questo serviva per predire le misure
        # human.measurement_loss(measurements, y_measures_test)
        mae, std = human.shape_parameters_loss(shape, y_shape_test)
        run['TEST_Shape Per Parameters Error-mae'].log(mae)
        run['TEST_Shape Per Parameters Error-std'].log(std)
        print(f"Shape Per Parameters Error: \n Mean Absolute Error +/- std : {np.round(mae, 5)} +/- {np.round(std, 5)}")
        mae, std, mape = human.per_vertex_shape_loss(p_verts, a_verts)
        print()
        print(f"3D shape per vertex Error: \n Mean Absolute Error +/- std : {np.round(mae, 5)} +/- {np.round(std, 5)}")
        print(f"Mape : {np.round(mape)}")
        run['TEST_3D shape per vertex Error-mae'].log(mae)
        run['TEST_3D shape per vertex Error-std'].log(std)
        run['TEST_Mape'].log(mape)


class PCAExtractor(object):
    def __init__(self, gender, loss):
        self.pca_front = load(f'weights3/pca_{gender}_front.joblib')
        self.pca_side = load(f'weights3/pca_{gender}_side.joblib')

        losses = {'mse': nn.MSELoss(), \
                  'mae': nn.L1Loss(), \
                  'mae+mse': MAASE(), \
                  'bce': nn.BCELoss()}
        self.loss = losses[loss]
        self.accuracy = Accuracy()

    def test_performance(self, data):
        front, side = data[:, :, :, 0], data[:, :, :, 1]
        side_data = side.reshape(data.shape[0], data.shape[1] * data.shape[2])
        front_data = front.reshape(data.shape[0], data.shape[1] * data.shape[2])

        start_time = time.time()

        front_features = self.pca_front.transform(front_data)
        side_features = self.pca_side.transform(side_data)
        recon_side_data = self.pca_side.inverse_transform(side_features)
        recon_front_data = self.pca_side.inverse_transform(front_features)
        recon_side_data = recon_side_data.reshape(len(data), 512, 512)
        recon_front_data = recon_front_data.reshape(len(data), 512, 512)

        front = torch.tensor(front).float()
        side = torch.tensor(side).float()
        recon_front_data = torch.tensor(recon_front_data).float()
        recon_side_data = torch.tensor(recon_side_data).float()

        front_loss = self.loss(front, recon_front_data).item()
        side_loss = self.loss(side, recon_side_data).item()
        front_accuracy = self.accuracy(front, recon_front_data).item()
        side_accuracy = self.accuracy(side, recon_side_data).item()

        elapsed_time = time.time() - start_time
        print(
            'Time={:.2f}s test_loss_front={:.4f} test_loss_side={:.4f} accuracy_front= {:.4f} accuracy_side= {:.4f}'.format(
                elapsed_time,
                front_loss, side_loss, front_accuracy, side_accuracy))

    def extract_features(self, data):
        front_data, side_data = data[:, :, :, 0], data[:, :, :, 1]
        side_data = side_data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        front_data = front_data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        front_features = self.pca_front.transform(front_data)
        side_features = self.pca_side.transform(side_data)
        features = np.concatenate([front_features, side_features], axis=-1)
        return features


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        DICE = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return DICE


class AEXtractor(object):
    def __init__(self, device, gender, batch_size, loss, loss2, path_folder, lr):
        self.device = device
        self.gender = gender
        self.batch_size = batch_size
        self.path_folder = path_folder

        losses = {'mse': nn.MSELoss(), \
                  'mae': nn.L1Loss(), \
                  'mae+mse': MAASE(), \
                  'bce': nn.BCELoss(), \
                  'dice': DiceLoss()}

        self.loss = losses[loss]
        self.loss2 = losses['bce']
        self.accuracy = Accuracy()

        feature_extractor_path = self.path_folder + '/weights3/' + lr.__str__() + '/new_base_feature_extractor_' + self.gender + '_' + args.epochs.__str__() + '.pth'
        feature_extractor_weights = torch.load(feature_extractor_path, map_location=torch.device(self.device))
        feature_extractor = Deep2DEncoder(image_size=512, kernel_size=3, n_filters=32)
        feature_extractor.load_state_dict(feature_extractor_weights)

        decoder_path = self.path_folder + '/weights3/' + lr.__str__() + '/new_base_decoder_' + self.gender + '_' + args.epochs.__str__() + '.pth'
        decoder_weights = torch.load(decoder_path, map_location=torch.device(self.device))
        decoder = Deep2DDecoder(image_size=512, kernel_size=3, n_filters=32)
        decoder.load_state_dict(decoder_weights)

        self.feature_extractor = feature_extractor.to(self.device)
        self.decoder = decoder.to(self.device)

    def test_performance(self, data_loader):

        feature_extractor = self.feature_extractor
        decoder = self.decoder

        reconstruction_criterion = self.loss

        epoch_loss_front = 0
        epoch_loss_side = 0
        total_accuracy_front = 0
        total_accuracy_side = 0

        feature_extractor.eval()
        decoder.eval()
        feature_extractor.requires_grad_(False)
        decoder.requires_grad_(False)

        step = 0
        start_time = time.time()

        for x_front, x_side in data_loader:
            x_front = x_front.to(device=self.device)
            x_side = x_side.to(device=self.device)
            feature_front = feature_extractor(x_front)
            feature_side = feature_extractor(x_side)

            x_hat_front = decoder(feature_front)
            prova = x_hat_front[0, 0, :, :].cpu().detach()
            # plt.imshow(prova)
            # plt.savefig(f'out_trainer{out}/{args.loader_type}/front_{step}.png')
            # imageio.imwrite(f'out_{out}/{args.loader_type}/front_{step}.png', prova)

            x_hat_side = decoder(feature_side)
            prova2 = x_hat_side[0, 0, :, :].cpu().detach()
            # plt.imshow(prova2)
            # plt.savefig(f'out_trainer{out}/{args.loader_type}/side_{step}.png')
            # imageio.imwrite(f'out_{out}/{args.loader_type}/side_{step}.png', prova2)

            total_accuracy_front += self.accuracy(x_hat_front, x_front)
            total_accuracy_side += self.accuracy(x_hat_side, x_side)

            recon_loss_front = reconstruction_criterion(x_hat_front, x_front)
            recon_loss_side = reconstruction_criterion(x_hat_side, x_side)
            recon_loss = recon_loss_front + recon_loss_side

            epoch_loss_front += recon_loss_front.item()
            epoch_loss_side += recon_loss_side.item()

            step += 1

        epoch_loss_front = epoch_loss_front / step
        epoch_loss_side = epoch_loss_side / step
        total_accuracy_front = total_accuracy_front / step
        total_accuracy_side = total_accuracy_side / step
        run[f'evaluator/{args.loader_type}/epoch_loss_front'].log(epoch_loss_front)
        run[f'evaluator/{args.loader_type}/epoch_loss_side'].log(epoch_loss_side)
        run[f'evaluator/{args.loader_type}/total_accuracy_front'].log(total_accuracy_front)
        run[f'evaluator/{args.loader_type}/total_accuracy_side'].log(total_accuracy_side)

        elapsed_time = time.time() - start_time
        string5 = "elapsed_time:"
        log5 = string5 + str(elapsed_time)
        logging.info(log5)
        print(
            'Time={:.2f}s test_loss_front={:.4f} test_loss_side={:.4f} accuracy_front= {:.4f} accuracy_side= {:.4f}'.format(
                elapsed_time,
                epoch_loss_front, epoch_loss_side, total_accuracy_front, total_accuracy_side))

    def extract_features(self, data):

        data = torch.tensor(data).float()
        front = data[:, :, :, 0]
        side = data[:, :, :, 1]

        front_features = []
        side_features = []

        self.feature_extractor.eval()
        # self.feature_extractor.train()
        self.feature_extractor.requires_grad_(False)
        # self.feature_extractor.requires_grad_(True)

        for f, s in tqdm(zip(front, side)):
            front_features.append(
                self.feature_extractor(f.view(1, 1, 512, 512).to(self.device)).cpu().detach().numpy().reshape(256))
            side_features.append(
                self.feature_extractor(s.view(1, 1, 512, 512).to(self.device)).cpu().detach().numpy().reshape(256))

        front_features = np.array(front_features)
        side_features = np.array(side_features)

        return np.concatenate([front_features, side_features], axis=-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1', help="GPU number")
    parser.add_argument("--path_folder", type=str, default='Dataset/data10/train_test_data_fold1',
                        help='Dataset/data10/train_test_data_fold1/')
    parser.add_argument("--data_path", type=str, default='dataloaders/male/train_512_images.npy')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gender", type=str, default='male')
    parser.add_argument("--model", type=str, default='ae', help='pca or ae')
    parser.add_argument("--lr", type=float, default=0.0001, help='pca or ae')
    parser.add_argument("--epochs", type=int, default=1, help='pca or ae')
    parser.add_argument("--mode", type=str, default='features', help='features or performance')
    parser.add_argument("--loss", type=str, default='dice', help='choose one: mae, mse, mae+mse, bce')

    args = parser.parse_args()


    print("seed di torch", torch.seed())
    folder = args.path_folder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Data(args.path_folder + '/dataloaders/' + args.gender + '/train_512_images.npy')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_dataset = Data(folder + '/dataloaders/' + args.gender + '/validation_512_images.npy')
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
    model = AETrainer(device=device, gpu=args.gpu, batch_size=args.batch_size, gender=args.gender, loss=args.loss,
                      loss2='bce')
    print("Training AE Started")
    model.train_model(dataloader, test_dataloader, epochs=args.epochs, args=args, learning_rate=args.lr, folder=folder)
    print("Training Done")
