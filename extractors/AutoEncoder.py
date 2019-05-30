import torch
import torchvision.datasets as datasets
import torchvision.transforms as Transform
from torch.optim import lr_scheduler
from PIL import Image
import torch.nn as nn
from torch.utils.data import DataLoader

import time
import re
import datetime
import os

from .base import FeatureExtractor


class AutoEncoderExtractor(FeatureExtractor):
    type = 'autoencoder'

    def __init__(self, params):
        self.input_shape = params['input_shape']
        self.pre_process = self.preprocess(self.input_shape)
        self.device = params['device']
        self.model = AutoEncoder(input_shape=self.input_shape)
        self.output_path = params['output_dir']
        if params['train']:
            self.train(params)
        elif params['model_path'] is not None and params['model_path'] != '':
            self.model.load_state_dict(torch.load(params['model_path'], map_location=lambda storage, location: storage))
            self.model.to(self.device)
        else:
            raise Exception('no model specified: use model_path to give path of trained model or set -train')
        self.model.eval()

    def get_training_helpers(self, params, model: nn.Module):
        num_epochs = params['epochs']
        milestone_ratio = params['milestone_step_ratio']
        learning_rate = params['learning_rate']
        weight_decay = params['weight_decay']
        gamma = params['gamma']
        loss = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(x * num_epochs) for x in milestone_ratio],
                                             gamma=gamma)
        return loss, optimizer, scheduler

    def preprocess(self, input_shape)->Transform:
        transform = Transform.Compose([Transform.Resize(input_shape[:2]),
                                       Transform.ToTensor(),
                                       Transform.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
        return transform

    def prepare_data(self, input_folder: str, pre_process: Transform, batch_size: int=64)->DataLoader:
        trainset = datasets.ImageFolder(input_folder, transform=pre_process)
        dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=4)
        return dataloader

    def train(self, params):
        print('training starts..')
        input_folder = params['input_train']
        input_shape = params['input_shape']
        batch_size = params['batch_size']
        num_epochs = params['epochs']
        model_directory = os.path.join(self.output_path, 'model')

        model = AutoEncoder(input_shape=input_shape)
        model = model.to(self.device)

        model.train(True)

        distance, optimizer, scheduler = self.get_training_helpers(params, model)
        data_loader = self.prepare_data(input_folder, self.pre_process, batch_size)

        for epoch in range(num_epochs):
            start_time = time.time()
            scheduler.step()
            epoch_loss = 0.0
            for data in data_loader:
                imgs, _ = data
                imgs = imgs.to(self.device)
                # ===================forward=====================
                encoded_outputs, decoded_outputs = model(imgs)
                loss = distance(decoded_outputs, imgs)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.data * imgs.size(0)
            # ===================log========================
            epoch_time = time.time() - start_time
            epoch_loss /= len(data_loader.dataset)
            if epoch % 100 == 0:
                print('saving checkpoint')
                time_string = re.sub(string='_'.join(str(datetime.datetime.now()).split(':')[:-1]), pattern='[ -]',
                                     repl='_')
                model_save_name = "model_" + str(epoch) + '_' + time_string + ".pt"
                torch.save(model.state_dict(),
                           os.path.join(model_directory, model_save_name))
            print('epoch [{}/{}], run_loss:{:.4f}, epoch_loss:{:.4f}, time:{:.4f}s'.format(epoch + 1, num_epochs,
                                                                                           loss.data, epoch_loss,
                                                                                           epoch_time))
        torch.save(model.state_dict(), os.path.join(model_directory, "model_final.pt"))
        self.model = model

    def extract_features(self, img: Image):
        img = self.pre_process(img).unsqueeze(0)
        img = img.to(self.device)
        feature_vector, _ = self.model(img)
        return feature_vector.cpu().detach().numpy().flatten()


class AutoEncoder(nn.Module):
    def __init__(self, input_shape=(224, 224, 3)):
        super(AutoEncoder, self).__init__()

        self.input_shape = input_shape

        self.encoder_conv = nn.Sequential(nn.Conv2d(input_shape[2], 32, kernel_size=5, stride=2, padding=2),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0),
                                          nn.ReLU(inplace=True))
        self.feature_reduction = (input_shape[0] // 2 // 2 // 2 // 2 - 1) // 2
        lin_features_len = ((input_shape[0] // 2 // 2 // 2 // 2 - 1) // 2) * (
                    (input_shape[0] // 2 // 2 // 2 // 2 - 1) // 2) * 128

        self.encoder_linear = nn.Linear(lin_features_len, 2048)

        self.decoder_layer = nn.Linear(2048, lin_features_len)

        out_pad = 1 if input_shape[0] // 2 // 2 // 2 // 2 % 2 == 0 else 0

        self.decoder_conv = nn.Sequential(nn.ReLU(inplace=True),
                                          nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=0,
                                                             output_padding=out_pad),
                                          nn.ReLU(inplace=True),
                                          nn.ConvTranspose2d(64, 64, kernel_size=5, stride=2, padding=2,
                                                             output_padding=out_pad),
                                          nn.ReLU(inplace=True),
                                          nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=2,
                                                             output_padding=out_pad),
                                          nn.ReLU(inplace=True),
                                          nn.ConvTranspose2d(32, 32, kernel_size=5, stride=2, padding=2,
                                                             output_padding=out_pad),
                                          nn.ReLU(inplace=True),
                                          nn.ConvTranspose2d(32, 3, kernel_size=5, stride=2, padding=2,
                                                             output_padding=out_pad))  # ,

    #                                       nn.ReLU(inplace=True),
    #                                       nn.Sigmoid())

    def forward(self, input):
        # encoder
        encoded_features = self.encoder_conv(input)
        encoded_features = encoded_features.view(encoded_features.size(0), -1)
        encoded = self.encoder_linear(encoded_features)

        # decoder
        decoded_linear = self.decoder_layer(encoded)
        decoded = decoded_linear.view(decoded_linear.size(0), 128, self.feature_reduction, self.feature_reduction)
        decoded = self.decoder_conv(decoded)
        return encoded, decoded

