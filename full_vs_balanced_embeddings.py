import torch
from torch.utils.data import DataLoader, TensorDataset
from load_data import load_training_data
from vae import VAE, train
import datetime
import time
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

#first instatiate a model to load the weights into
LAYERS = [1000, 800, 600, 400, 200, 100] #based on hyperparameter tuning
LATENT = 10
device = 'cpu'
testModel = VAE(layers_data=LAYERS.copy(), input_dim=1422, latent_dim=LATENT).to(device)

#then load the weights
testModel.load_state_dict(torch.load("/Users/graha880/Desktop/mbVAE_resources/trained_models/subsampled_model.pt"))

#now embed the training data into the model
train_dataset = pd.read_csv("/Users/graha880/Desktop/mbVAE_resources/subsampled_geographic_train_data.csv", sep=',')
train_dataset = torch.tensor(train_dataset.to_numpy(),dtype=torch.float32)

#the encode function returns both the mean and logvar tensors, for this purpose we only want mean
training_data_embeddings = testModel.encode(train_dataset)[0]
training_data_embeddings = pd.DataFrame(training_data_embeddings.detach().numpy())
training_data_embeddings.to_csv('/Users/graha880/Desktop/mbVAE_resources/embeddings/subsample_train_data_embeddings.csv')

#now embed the test data into the model
test_dataset = pd.read_csv("/Users/graha880/Desktop/mbVAE_resources/geographic_test_data.csv")
test_dataset = torch.tensor(test_dataset.to_numpy(),dtype=torch.float32)

#the encode function returns both the mean and logvar tensors, for this purpose we only want mean
test_data_embeddings = testModel.encode(test_dataset)[0]
test_data_embeddings = pd.DataFrame(test_data_embeddings.detach().numpy())
test_data_embeddings.to_csv('/Users/graha880/Desktop/mbVAE_resources/embeddings/subsample_test_data_embeddings.csv')
