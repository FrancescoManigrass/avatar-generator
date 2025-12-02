import argparse
import os
import torch.nn as nn
import torch
import tqdm as tqdm
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
from joblib import load
from sklearn.kernel_ridge import KernelRidge
import logging
import neptune.new as neptune
import imageio

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

#lr = '0.0001_3'
lr = '10'
out = '2'


class PCAExtractor(object):
    def __init__(self, gender, loss):
        self.pca_front = load(f'weights/pca_{gender}_front.joblib')
        self.pca_side = load(f'weights/pca_{gender}_side.joblib')
        
        losses = {'mse' : nn.MSELoss(),\
                  'mae' : nn.L1Loss(),\
                  'mae+mse': MAASE(),\
                  'bce': nn.BCELoss()}
        self.loss = losses[loss]
        self.accuracy = Accuracy()

    def test_performance(self,data):
        front, side = data[:,:,:,0], data[:,:,:,1]
        side_data = side.reshape(data.shape[0], data.shape[1]*data.shape[2])
        front_data = front.reshape(data.shape[0], data.shape[1]*data.shape[2])

        start_time = time.time()

        front_features = self.pca_front.transform(front_data)
        side_features = self.pca_side.transform(side_data)
        recon_side_data = self.pca_side.inverse_transform(side_features)
        recon_front_data = self.pca_side.inverse_transform(front_features)
        recon_side_data = recon_side_data.reshape(len(data), 512,512)
        recon_front_data = recon_front_data.reshape(len(data), 512,512)

        front = torch.tensor(front).float()
        side = torch.tensor(side).float()
        recon_front_data = torch.tensor(recon_front_data).float()
        recon_side_data = torch.tensor(recon_side_data).float()
        
        front_loss = self.loss(front, recon_front_data).item()
        side_loss = self.loss(side, recon_side_data).item()
        front_accuracy = self.accuracy(front, recon_front_data).item()
        side_accuracy = self.accuracy(side, recon_side_data).item()

        elapsed_time = time.time() - start_time
        print('Time={:.2f}s test_loss_front={:.4f} test_loss_side={:.4f} accuracy_front= {:.4f} accuracy_side= {:.4f}'.format(
                                elapsed_time, 
                                front_loss, side_loss, front_accuracy, side_accuracy))    
 
    def extract_features(self,data):
        front_data, side_data = data[:,:,:,0], data[:,:,:,1]
        side_data = side_data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        front_data = front_data.reshape(data.shape[0], data.shape[1]*data.shape[2])
        front_features = self.pca_front.transform(front_data)
        side_features = self.pca_side.transform(side_data)
        features = np.concatenate([front_features, side_features], axis = -1)
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
    def __init__(self, device, gender, batch_size, loss, loss2,path_folder,lr):
        self.device = device
        self.gender = gender
        self.batch_size = batch_size
        self.path_folder=path_folder

        losses = {'mse' : nn.MSELoss(),\
                  'mae' : nn.L1Loss(),\
                  'mae+mse': MAASE(),\
                  'bce': nn.BCELoss(),\
                  'dice': DiceLoss()}
        
        self.loss = losses[loss]
        self.loss2 = losses['bce']
        self.accuracy = Accuracy()

        feature_extractor_path =  self.path_folder+'/weights/'+lr.__str__()+'/new_base_feature_extractor_'+self.gender+'_1000.pth'
        feature_extractor_weights = torch.load(feature_extractor_path,  map_location=torch.device(self.device))
        feature_extractor = Deep2DEncoder(image_size= 512 , kernel_size=3, n_filters=32)
        feature_extractor.load_state_dict(feature_extractor_weights)

        decoder_path = self.path_folder+'/weights/'+lr.__str__()+'/new_base_decoder_'+self.gender+'_1000.pth'
        decoder_weights = torch.load(decoder_path,  map_location=torch.device(self.device))
        decoder = Deep2DDecoder(image_size= 512 , kernel_size=3, n_filters=32)
        decoder.load_state_dict(decoder_weights)

        self.feature_extractor = feature_extractor.to(self.device)
        self.decoder = decoder.to(self.device)

    def test_performance(self, data_loader):

        run = neptune.init_run(
        project="GRAINS/3DAvatarGenerator",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZmI1MzQwMS0xZTdiLTRkZWQtOGQ4Yi0yNWE0ZGQ1MzEzNTQifQ==",
        )

        hyperparameters = {'model': 'AE', 'mode': 'features', 'loss': 'bce', 'batch_size': '16', 'loader_type': args.loader_type}                   
        run[f'evaluator/{args.loader_type}/hyperparameters']=hyperparameters

        feature_extractor = self.feature_extractor
        decoder = self.decoder

        reconstruction_criterion = self.loss
    
        epoch_loss_front= 0
        epoch_loss_side = 0
        total_accuracy_front = 0
        total_accuracy_side = 0

        feature_extractor.eval()
        decoder.eval()
        feature_extractor.requires_grad_(False)
        decoder.requires_grad_(False)

        step=0
        start_time = time.time()
        
        for x_front, x_side in data_loader:
            x_front = x_front.to(device=self.device)
            x_side = x_side.to(device=self.device)
            feature_front = feature_extractor(x_front)
            feature_side = feature_extractor(x_side)

            x_hat_front = decoder(feature_front)
            prova=x_hat_front[0,0,:,:].cpu().detach()
            #plt.imshow(prova)
            #plt.savefig(f'out_trainer{out}/{args.loader_type}/front_{step}.png')
            #imageio.imwrite(f'out_{out}/{args.loader_type}/front_{step}.png', prova)

            x_hat_side = decoder(feature_side)
            prova2=x_hat_side[0,0,:,:].cpu().detach()
            #plt.imshow(prova2)
            #plt.savefig(f'out_trainer{out}/{args.loader_type}/side_{step}.png')
            #imageio.imwrite(f'out_{out}/{args.loader_type}/side_{step}.png', prova2)
            
            total_accuracy_front += self.accuracy(x_hat_front, x_front)
            total_accuracy_side += self.accuracy(x_hat_side, x_side)

            recon_loss_front = reconstruction_criterion(x_hat_front, x_front)
            recon_loss_side = reconstruction_criterion(x_hat_side, x_side)
            recon_loss = recon_loss_front+recon_loss_side

            epoch_loss_front += recon_loss_front.item()
            epoch_loss_side += recon_loss_side.item()
            
            step+=1
        
        epoch_loss_front = epoch_loss_front / step
        epoch_loss_side = epoch_loss_side / step
        total_accuracy_front = total_accuracy_front / step
        total_accuracy_side = total_accuracy_side / step
        run[f'evaluator/{args.loader_type}/epoch_loss_front'].log(epoch_loss_front)
        run[f'evaluator/{args.loader_type}/epoch_loss_side'].log(epoch_loss_side)
        run[f'evaluator/{args.loader_type}/total_accuracy_front'].log(total_accuracy_front)
        run[f'evaluator/{args.loader_type}/total_accuracy_side'].log(total_accuracy_side)



        elapsed_time = time.time() - start_time 
        string5="elapsed_time:"
        log5=string5+str(elapsed_time)
        logging.info(log5)
        print('Time={:.2f}s test_loss_front={:.4f} test_loss_side={:.4f} accuracy_front= {:.4f} accuracy_side= {:.4f}'.format(
                                elapsed_time,
                                epoch_loss_front, epoch_loss_side, total_accuracy_front, total_accuracy_side))
        
    
    def extract_features(self, data):
        
        data = torch.tensor(data).float()
        front = data[:, :, :, 0]
        side = data[:, :, :, 1]

        front_features = []
        side_features = []
        
        self.feature_extractor.eval()
        #self.feature_extractor.train()
        self.feature_extractor.requires_grad_(False)
        #self.feature_extractor.requires_grad_(True)

        for f,s in tqdm.tqdm(zip(front, side)):
            front_features.append(self.feature_extractor(f.view(1, 1, 512, 512)).detach().numpy().reshape(256))
            side_features.append(self.feature_extractor(s.view(1, 1, 512, 512)).detach().numpy().reshape(256))
        
        front_features = np.array(front_features)
        side_features = np.array(side_features)

        return np.concatenate([front_features, side_features], axis = -1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", type=str, default='_512_images.npy')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gender", type=str, default='male')
    parser.add_argument("--model", type=str, default='ae', help='pca or ae')
    parser.add_argument("--mode", type=str, default='features', help='features or performance')
    parser.add_argument("--loss", type=str, default='bce', help='choose one: mae, mse, mae+mse, bce')
    parser.add_argument("--dataset", type=str, default='calvis', help='calvis or nomo')
    parser.add_argument("--loader_type", type = str, required = True, help='train or test or validation')
    parser.add_argument("--lr", type=float, required=True, help='learning rate')
    parser.add_argument("--path_folder", type=str, default='Dataset/data10/train_test_data_fold1')


    args = parser.parse_args()


    data_path=args.path_folder+"/"+args.loader_type+args.data_path
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    
    if args.model == 'pca':
        mode = args.data_path.split("/")[-1].split('_')[0]
        data = np.load(args.data_path)
        data = np.array(data,dtype='float')/255.0
        data[data < 1] = 0
        data = 1 - data

        pca = PCAExtractor(gender = args.gender, loss = args.loss)
        
        if  args.mode == 'performance':
            
            print('PCA Evaluation Started ')
            pca.test_performance(data)
            print('Evaluation Done ')
        
        else:
            
            print('Extracting PCA Features .. ')
            features = pca.extract_features(data)
            
            if args.dataset == 'calvis':
                np.save(f"data/dataloaders/{args.gender}_new/{args.model}_{args.loader_type}_features.npy", features)
            else:
                np.save(path_folder+f"/{args.gender}/{args.model}_{args.loader_type}_features.npy", features)

            print('Extraction Done ')

    if args.model == 'ae':
        model = AEXtractor(device=device, batch_size=args.batch_size, gender = args.gender, loss = args.loss, loss2='bce',path_folder=args.path_folder,lr=args.lr)

        if args.mode == 'performance':

            dataset = Data(args.data_path)
            dataloader = DataLoader(dataset, batch_size = args.batch_size, shuffle=True, drop_last=True,num_workers=8)
            print("AE Evaluation Started")
            model.test_performance(dataloader)
            print("Evaluation Done")
        
        else:
            mode = args.data_path.split("/")[-1].split('_')[0]
            data = np.load(args.path_folder+"/dataloaders/"+args.gender+"/"+args.loader_type+args.data_path)
            data = np.array(data,dtype='float16')/255.0
            data[data < 1] = 0
            data = 1 - data

            print("Extracting Encoder Features")
            features = model.extract_features(data)
            if args.dataset == 'calvis':
                np.save(f"ae_krr/10/base/{args.gender}/new_base_male_{args.model}_{args.loader_type}_features.npy", features)
            else:
                os.makedirs(args.path_folder+"/features"+"/"+args.gender,exist_ok=True)
                np.save(f"{args.path_folder}/features/{args.gender}/{args.model}_{args.loader_type}_features.npy", features)

                
            print("Extraction Done")
