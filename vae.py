#import time
import pandas as pd
import torch
#import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
#import anndata as ad
#import scanpy as sc
#import torchvision.transforms as transforms
#from torchvision.utils import save_image, make_grid
from loss import ZINBLoss, MeanAct, DispAct
from load_data import combined_training_set, raw_train_dataset, test_dataset
#from sklearn.preprocessing import StandardScaler

# create a transform to apply to each datapoint
#transform = transforms.Compose([transforms.ToTensor()])


# create train and test dataloaders
batch_size = 100
train_loader = DataLoader(dataset=combined_training_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class VAE(nn.Module):

	def __init__(self, layers_data=[],input_dim=1422, latent_dim=5,device='cpu'):  # input dim is number of taxa
		super(VAE, self).__init__()
		self._layers_data = layers_data
		self._latent_dim = latent_dim
		# encoder
		#self.encoder = nn.Sequential(
		#	nn.Linear(input_dim, hidden_dim),
		#	nn.LeakyReLU(0.2)
		#)
		#create the encoder; start it as an empty moduleList that we iterate through to add to layers
		#encoder will have as many hidden layers as elements in layers_data; starting with input size and ending with latent size
		self.encoder = nn.ModuleList()
		prev_size = input_dim #first layer will go from input size to current size
		for curr_size in layers_data:
			self.encoder.append(nn.Linear(prev_size, curr_size))
			self.encoder.append(nn.LeakyReLU(0.2))
			prev_size = curr_size
		self._enc_mu = nn.Linear(layers_data[-1], latent_dim)
		self._enc_logvar = nn.Linear(layers_data[-1], latent_dim)

		# latent mean and variance
		#self.mean_layer = nn.Linear(latent_dim, 2)
		#self.logvar_layer = nn.Linear(latent_dim, 2)

		# decoder
		#self.decoder = nn.Sequential(
		#	nn.Linear(latent_dim, hidden_dim),
		#	nn.LeakyReLU(0.2),
		#)
		#create the decoder; start it as an empty moduleList that we iterate through to add layers
		self.decoder = nn.ModuleList()
		layers_data.reverse() #for the decoder, we need to iterate in the opposite order of input)
		prev_size = latent_dim #first layer will go from input size to current size
		for curr_size in layers_data:
			self.decoder.append(nn.Linear(prev_size,curr_size))
			self.decoder.append(nn.LeakyReLU(0.2))
			prev_size = curr_size

		self._dec_mean = nn.Sequential(nn.Linear(layers_data[-1], input_dim), MeanAct())
		self._dec_disp = nn.Sequential(nn.Linear(layers_data[-1], input_dim), DispAct())
		self._dec_pi = nn.Sequential(nn.Linear(layers_data[-1], input_dim), nn.Sigmoid())
		# loss function
		self.zinb_loss = ZINBLoss()

	def encode(self, x):
		#encoder is a moduleList, now have to create the actual layers
		for layer in self.encoder:
			x = layer(x)
		#x = self.encoder(x)
		mean, logvar = self._enc_mu(x), self._enc_logvar(x)
		return mean, logvar

	def reparameterization(self, mean, var):
		epsilon = torch.randn_like(var).to(device)
		z = mean + var * epsilon
		return z

	def decode(self, x):
		for layer in self.decoder:
			x = layer(x)
		return x

	def forward(self, x):
		latent_mean, logvar = self.encode(x) # get the mean and log variance of the latent space
		z = self.reparameterization(latent_mean, logvar) # reparameterization trick
		x_hat = self.decode(z) # get the decoded output
		_mean = self._dec_mean(x_hat) # output of initial size
		_disp = self._dec_disp(x_hat) # dispersion estimate
		_pi = self._dec_pi(x_hat) # pi estimate
		return latent_mean, logvar, _mean, _disp, _pi

	def reconstruct(self, x):
		encoded = self.encode(x)[0]
		decoded = self.decode(encoded)
		return self._dec_mean(decoded)
		#return self.decode(self.encode(x))


def KLD(mean, log_var):
	KLD = - 0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
	return KLD

layers = [1000,700,400,100] #based on hyperparameter tuning
latent = 10
epochs = 20
lr = 1e-5

#model = VAE().to(device)  # this switches the data/model/whatever to whatever device you want to work on (eg from CPU memory to a GPU)
model = VAE(layers_data=layers.copy(),input_dim=1422,latent_dim=latent).to(device) #need to move input dim to end of args but i'm too lazy rn
#optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Adam is an algorithm for stochastic optimization
optimizer = torch.optim.Adam(model.parameters(),lr=lr)
def train(model,optimizer, epochs, device,features,data): #data in the format of a dataLoader
	model.train() # set model to training mode
	for epoch in range(epochs):
		overall_loss = 0
		#for batch_idx, x in enumerate(train_loader):
		for batch_idx, (norm_data, raw_data, batch_sf) in enumerate(data):
			x_norm = norm_data.view(batch_size,features).to(device)
			x_raw = raw_data.view(batch_size,features).to(device)
			#batch_sf = batch_sf.view(batch_size,features).to(device)
			optimizer.zero_grad()
			latent_mean, latent_logvar, mean, disp, pi = model(x_norm)
			KLD_loss = KLD(latent_mean,latent_logvar)
			zinb_loss = model.zinb_loss(x_raw,mean, disp, pi)
			loss = KLD_loss + zinb_loss
			overall_loss += loss.item()

			loss.backward()
			optimizer.step()

		print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss / (batch_idx * batch_size))
	return overall_loss


#output = train(model, optimizer, epochs=20, device=device,features=raw_train_dataset.shape[1],data=train_loader)
#test_output = model.reconstruct(raw_train_dataset)

#latent_mean, latent_logvar, mean, disp, pi = model(norm_train_dataset)

#normInput = norm_train_dataset.detach().numpy()
#normInput = pd.DataFrame(normInput)
#normInput.to_csv('~/Desktop/mbVAE/zinb_normalized_input.csv')


#meanNP = mean.detach().numpy()
#meanNP = pd.DataFrame(meanNP)
#meanNP.to_csv('~/Desktop/mbVAE/zinb_mean_output.csv')

#inputNP = input.detach().numpy()
#inputNP = pd.DataFrame(inputNP)
#inputNP.to_csv('~/Desktop/mbVAE/zinb_input_abrv.csv')

#t_np = t.numpy() #convert to Numpy array

#latentNP = latent_mean.detach().numpy()
#latentNP = pd.DataFrame(latentNP)
#latentNP.to_csv('~/Desktop/mbVAE/latent_mean.csv')

#df = pd.DataFrame(t_np) #convert to a dataframe
#df.to_csv("testfile",index=False) #save to file
