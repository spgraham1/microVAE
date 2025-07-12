import torch
from torch.utils.data import DataLoader, TensorDataset
from load_data import load_training_data
from vae import VAE, train
import datetime
import time
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


FEATURES = 1422
batch_size = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYERS = [1000, 800, 600, 400, 200, 100] #based on hyperparameter tuning
LATENT = 10
EPOCHS = 20
lr = 1e-5

#fullDatasetModel = VAE(layers_data=LAYERS.copy(),input_dim=1422,latent_dim=LATENT).to(device)
#subDatasetModel = VAE(layers_data=LAYERS.copy(),input_dim=1422,latent_dim=LATENT).to(device)
#optimizer = torch.optim.Adam(fullDatasetModel.parameters(),lr=lr)

datafiles = ["/Users/graha880/Desktop/mbVAE_resources/50_geographic_train_data.csv","/Users/graha880/Desktop/mbVAE_resources/100_geographic_train_data.csv","/Users/graha880/Desktop/mbVAE_resources/300_geographic_train_data.csv","/Users/graha880/Desktop/mbVAE_resources/600_geographic_train_data.csv", "/Users/graha880/Desktop/mbVAE_resources/subsampled_geographic_train_data.csv",'/Users/graha880/Desktop/mbVAE_resources/full_geographic_train_data.csv']


#datafiles = ['/Users/graha880/Desktop/mbVAE_resources/full_geographic_train_data.csv']

#should later update the loading here to use the function in load_data
for infile in datafiles:
	train_dataset = pd.read_csv(infile, sep=',')
	sample_counts = train_dataset.sum(axis=1)
	size_factor = sample_counts / np.median(sample_counts)
	raw = train_dataset.to_numpy()  # get the raw training data
	scaler = StandardScaler()
	norm = scaler.fit_transform(train_dataset)

	# convert all to tensors
	raw_train_dataset = torch.tensor(raw, dtype=torch.float32)
	norm_train_dataset = torch.tensor(norm, dtype=torch.float32)
	size_factor = torch.tensor(size_factor, dtype=torch.float32)

	combined_training_set = TensorDataset(norm_train_dataset, raw_train_dataset, size_factor)


	date = datetime.date.today().strftime("%Y-%m-%d")
	type = infile.replace('/Users/graha880/Desktop/mbVAE_resources/','')
	type = type.replace('_geographic_train_data.csv','')
	print(type)
	outFile = '/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/reconstruction/' + type + '_output_' + date + '.csv'
	modelOutFile = '/Users/graha880/Desktop/mbVAE_resources/trained_models/' + type + '_model' + '.pt'
	print(outFile)
	print(modelOutFile)
	start_time = time.perf_counter()
	print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
	model = VAE(layers_data=LAYERS.copy(), input_dim=1422, latent_dim=LATENT).to(
		device)  # need to move input dim to end of args but i'm too lazy rn
	train_loader = DataLoader(dataset=combined_training_set, batch_size=batch_size, shuffle=True,
							  drop_last=True)
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	total_loss = train(model,optimizer, EPOCHS, device,FEATURES,train_loader) #data in the format of a dataLoader
	print('loss during training: ' + str(total_loss))
	model.eval() #set model to evaluation mode for reconstrction
	with torch.no_grad():
		output = model.reconstruct(raw_train_dataset)
		output = output.detach().numpy()
		output = pd.DataFrame(output)
		output.to_csv(outFile)
		print('shape: ' + str(output.shape))
	torch.save(model.state_dict(),modelOutFile)

	end_time = time.perf_counter()
	lapsed = end_time - start_time
	print('lapsed seconds: ' + str(lapsed))




fullTensor = load_training_data('/Users/graha880/Desktop/mbVAE_resources/full_geographic_train_data.csv')


fullDataLoader = DataLoader(dataset=fullTensor, batch_size=batch_size, shuffle=True,drop_last=True)
subsetTensor = load_training_data('/Users/graha880/Desktop/mbVAE_resources/subsampled_geographic_train_data.csv')
subsetLoader = DataLoader(dataset=subsetTensor, batch_size=batch_size, shuffle=True,drop_last=True)

subsetLoss = train(subDatasetModel,optimizer,epochs=EPOCHS,device=device,features=FEATURES,data=subsetLoader)
print('subset dataset loss:')
print(subsetLoss)

fullLoss = train(fullDatasetModel,optimizer,epochs=EPOCHS,device=device,features=FEATURES,data=fullDataLoader)
print('full dataset loss:')
print(fullLoss)



