import argparse
import os
from os.path import join

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
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



def cc(image):
    for i in range(16):
      image=image[i,0,:,:]
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

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        DICE = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
                
        return DICE


class AETrainer(object):
    def __init__(self, device, gpu, gender, batch_size, loss, loss2):
        self.device = device
        self.gender = gender
        self.gpu = gpu
        self.batch_size = batch_size

        losses = {'mse' : nn.MSELoss(),\
                  'mae' : nn.L1Loss(),\
                  'mae+mse': MAASE(),\
                  'bce': nn.BCELoss(),\
                  'diceBCE': DiceBCELoss(),\
                  'dice': DiceLoss()}
        
        self.loss = losses[loss]
        self.loss2 = losses['bce']
        self.epochs = None
        self.accuracy = Accuracy()

    def train_model(self, data_loader, test_data_loader,args,epochs = 1000, learning_rate= 0.0001, betas = (0.9, 0.99),folder=""):

        lr="0.0001"

        run = neptune.init_run(
        project="GRAINS/3DAvatarGenerator",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZmI1MzQwMS0xZTdiLTRkZWQtOGQ4Yi0yNWE0ZGQ1MzEzNTQifQ==",
        )

        """

        run = neptune.init_run(
        project="federica.moro/3DAvatarGenerator",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZmI1MzQwMS0xZTdiLTRkZWQtOGQ4Yi0yNWE0ZGQ1MzEzNTQifQ==",

        )
        """

        hyperparameters = {'n_epochs':epochs, 'learning_rate':learning_rate, 'model': 'AE', 'loss': 'bce', 'batch_size': args.batch_size.__str__()}
        run['hyperparameters']=hyperparameters
        run["config_files"].upload_files([os.path.join(dp, f) for dp, dn, filenames in os.walk(".") for f in filenames if os.path.splitext(f)[1] == '.py'])

        
        # image_size is the size of input mask
        self.epochs = epochs

        
        #feature_extractor = Deep2DEncoder(image_size= 512 , kernel_size=3, n_filters=32)
        #feature_extractor_path = f'weights/0.0001/feature_extractor_male_1000.pth'
        #feature_extractor_weights = torch.load(feature_extractor_path, map_location=torch.device(self.device))
        #feature_extractor.load_state_dict(feature_extractor_weights)
        #decoder = Deep2DDecoder(image_size=512, kernel_size=3, n_filters=32)
        #decoder_path = f'weights/0.0001/decoder_male_1000.pth'
        #decoder_weights = torch.load(decoder_path, map_location=torch.device(self.device))
        #decoder.load_state_dict(decoder_weights)
                
        feature_extractor = Deep2DEncoder(image_size= 512 , kernel_size=3, n_filters=32)
        decoder = Deep2DDecoder(image_size=512, kernel_size=3, n_filters=32)

        feature_extractor.to(self.device)
        decoder.to(self.device)

        optimizer_extractor = torch.optim.Adam(feature_extractor.parameters(), lr = learning_rate, betas= betas)
        optimizer_decoder = torch.optim.Adam(decoder.parameters(), lr = learning_rate, betas= betas)

        #optimizer_extractor_weights = torch.load(f'weights/0.0001/extractor_opt_{self.gender}_1000.pth', map_location=torch.device(self.device))
        #optimizer_decoder_weights = torch.load(f'weights/0.0001/decoder_opt_{self.gender}_1000.pth', map_location=torch.device(self.device))
        #optimizer_extractor.load_state_dict(optimizer_extractor_weights)
        #optimizer_decoder.load_state_dict(optimizer_decoder_weights)

        ##reconstruction_criterion1 = self.loss
        reconstruction_criterion2 = self.loss2

        cuda = True if torch.cuda.is_available() else False
        
        if cuda:
            ##reconstruction_criterion1.cuda()
            reconstruction_criterion2.cuda()
        
        start_time = time.time()

        for epoch in range(epochs):

            # TRAINING
            print("inizio nuova epoch")
            
            feature_extractor.train()
            decoder.train()
            feature_extractor.requires_grad_(True)
            decoder.requires_grad_(True)
            
            step=0
            total_accuracy_front = 0
            total_accuracy_side = 0
            total_recon_loss_dice = 0
            total_recon_loss_bce = 0 
            total_recon_loss = 0

            for x_front, x_side in data_loader:
                x_front = x_front.to(device=self.device)
                x_side = x_side.to(device=self.device)
                feature_front = feature_extractor(x_front)
                feature_side = feature_extractor(x_side)

                x_hat_front = decoder(feature_front)
                x_hat_side = decoder(feature_side)

                #print(x_hat_front.shape)
                #print(x_hat_front)

                # connected components su x_hat_front e x_hat_side
                # ottengo l'immagine nuova sulla quale applico la dice loss
                #mask_front = cc(x_hat_front)
                #mask_side = cc(x_hat_side)

                # commentare nel caso di utilizzo di modello preallenato
                ##recon_loss_front_dice = reconstruction_criterion1(x_hat_front, x_front)
                ##recon_loss_side_dice = reconstruction_criterion1(x_hat_side, x_side)
                recon_loss_front_bce = reconstruction_criterion2(x_hat_front, x_front)
                recon_loss_side_bce = reconstruction_criterion2(x_hat_side, x_side)


                # nuova recon_loss solo con la dice
                #recon_loss_front = reconstruction_criterion(mask_front, x_front)
                #recon_loss_side = reconstruction_criterion(mask_side, x_side)
                ##recon_loss_dice = recon_loss_front_dice+recon_loss_side_dice
                recon_loss_bce = recon_loss_front_bce+recon_loss_side_bce

                #recon_loss = recon_loss_dice+(recon_loss_bce*0.5)
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
                
                step+=1

                #run['TRAIN accuracy front'].log(self.accuracy(x_hat_front, x_front))
                #run['TRAIN accuracy side'].log(self.accuracy(x_hat_side, x_side))
                ##run['TRAIN recon_loss_dice'].log(recon_loss_dice.item())
                #run['TRAIN recon_loss_bce'].log(recon_loss_bce.item())
                #run['TRAIN recon_loss'].log(recon_loss.item()) 

             

            # TESTING
            feature_extractor.eval()
            decoder.eval()
            feature_extractor.requires_grad_(False)
            decoder.requires_grad_(False)

            step_test=0
            total_accuracy_front_test = 0
            total_accuracy_side_test = 0
            total_recon_loss_dice_test = 0
            total_recon_loss_bce_test = 0 
            total_recon_loss_test = 0

            for x_front, x_side in test_data_loader:
                x_front = x_front.to(device=self.device)
                x_side = x_side.to(device=self.device)
                feature_front = feature_extractor(x_front)
                feature_side = feature_extractor(x_side)

                x_hat_front = decoder(feature_front)
                x_hat_side = decoder(feature_side)

                #print(x_hat_front.shape)
                #print(x_hat_front)
                #mask_front = cc(x_hat_front)
                #mask_side = cc(x_hat_side)

                #recon_loss_front = reconstruction_criterion(x_hat_front, x_front)
                #recon_loss_side = reconstruction_criterion(x_hat_side, x_side)

                ##recon_loss_front_dice = reconstruction_criterion1(x_hat_front, x_front)
                ##recon_loss_side_dice = reconstruction_criterion1(x_hat_side, x_side)
                recon_loss_front_bce = reconstruction_criterion2(x_hat_front, x_front)
                recon_loss_side_bce = reconstruction_criterion2(x_hat_side, x_side)

                ##recon_loss_dice = recon_loss_front_dice+recon_loss_side_dice
                recon_loss_bce = recon_loss_front_bce+recon_loss_side_bce

                ##recon_loss = recon_loss_dice+(recon_loss_bce*0.5)
                recon_loss = recon_loss_bce


                total_accuracy_front_test += self.accuracy(x_hat_front, x_front)
                total_accuracy_side_test += self.accuracy(x_hat_side, x_side)
                ##total_recon_loss_dice_test += recon_loss_dice.item()
                total_recon_loss_bce_test += recon_loss_bce.item()
                total_recon_loss_test += recon_loss.item()
                
                step_test+=1

                #run['TEST accuracy front'].log(self.accuracy(x_hat_front, x_front))
                #run['TEST accuracy side'].log(self.accuracy(x_hat_side, x_side))
                ##run['TEST recon_loss_dice'].log(recon_loss_dice.item())
                #run['TEST recon_loss_bce'].log(recon_loss_bce.item())
                #run['TEST recon_loss'].log(recon_loss.item())


            #
            elapsed_time = time.time() - start_time 
            # print('Epoch {}/{} Time={:.2f}s train_loss_front={:.4f} train_loss_side={:.4f} accuracy_front= {:.4f} accuracy_side= {:.4f}'.format(
            #                         epoch +1, epochs,
            #                         elapsed_time,
            #                         epoch_loss_front, epoch_loss_side, total_accuracy_front, total_accuracy_side))
            print('Epoch {}/{} Time={:.2f}s'.format(epoch +1, epochs, elapsed_time))

            run['Total TRAIN accuracy front'].log(total_accuracy_front/step)
            run['Total TRAIN accuracy side'].log(total_accuracy_front/step)
            ##run['Total TRAIN recon_loss_dice'].log(total_recon_loss_dice/step)
            ##run['Total TRAIN recon_loss_bce'].log(total_recon_loss_bce/step)
            run['Total TRAIN recon_loss'].log(total_recon_loss/step)

            run['Total TEST accuracy front'].log(total_accuracy_front_test/step_test)
            run['Total TEST accuracy side'].log(total_accuracy_side_test/step_test)
            ##run['Total TEST recon_loss_dice'].log(total_recon_loss_dice_test/step_test)
            ##run['Total TEST recon_loss_bce'].log(total_recon_loss_bce_test/step_test)
            run['Total TEST recon_loss'].log(total_recon_loss_test/step_test)

            os.makedirs(folder+"/"+f'weights/{lr}',exist_ok=True)

            torch.save(feature_extractor.state_dict(), folder+"/"+ f'weights/{lr}/new_base_feature_extractor_{self.gender}_{epochs}.pth')
            torch.save(decoder.state_dict(), folder+"/"+ f'weights/{lr}/new_base_decoder_{self.gender}_{epochs}.pth')
            torch.save(optimizer_extractor.state_dict(), folder+"/"+f'weights/{lr}/new_base_extractor_opt_{self.gender}_{epochs}.pth')
            torch.save(optimizer_decoder.state_dict(), folder+"/"+f'weights/{lr}/new_base_decoder_opt_{self.gender}_{epochs}.pth')

        
        try:
            os.mkdir("weights")
        except:
            pass


#Seed=42
#torch.manual_seed(Seed)
    

if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default='1', help="GPU number")
    parser.add_argument("--path_folder", type=str, default='Dataset/data10/train_test_data_fold1/', help='Dataset/data10/train_test_data_fold1/')
    parser.add_argument("--data_path", type=str, default='dataloaders/male/train_512_images.npy')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gender", type=str, default='male')
    parser.add_argument("--loss", type=str, default='dice', help='choose one: mae, mse, mae+mse, bce')



    args = parser.parse_args()

    print("seed di torch",torch.seed())
    folder = args.path_folder
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = Data(join(args.path_folder,args.data_path))
    dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True, drop_last=True,num_workers=8)
    test_dataset = Data(folder+'dataloaders/'+args.gender+'/validation_512_images.npy')
    test_dataloader = DataLoader(test_dataset, batch_size = args.batch_size, shuffle=False, drop_last=True,num_workers=8)
    model = AETrainer(device=device, gpu=args.gpu, batch_size=args.batch_size, gender = args.gender, loss = args.loss, loss2 = 'bce')
    print("Training AE Started")
    model.train_model(dataloader, test_dataloader, epochs=1000,args=args,folder=folder)
    print("Training Done")
