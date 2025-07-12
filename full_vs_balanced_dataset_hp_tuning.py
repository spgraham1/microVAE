from vae import VAE, KLD
from load_data import combined_training_set, raw_train_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
import datetime
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


#train_loader = DataLoader(dataset=combined_training_set, batch_size=batch_size, shuffle=True)
device='cpu'

def tune_hyperparameters(model,lr, optimizer, epochs, layers_data, latent_size, outFile, device,features,batch_size,training_data):
	model.train() # set model to training mode
	for epoch in range(epochs):
		overall_loss = 0
		KLDsave = 0
		zinbsave = 0
		#for batch_idx, x in enumerate(train_loader):
		for batch_idx, (norm_data, raw_data, batch_sf) in enumerate(training_data):
			x_norm = norm_data.view(batch_size,features).to(device)
			x_raw = raw_data.view(batch_size,features).to(device)
			batch_sf = batch_sf.view(batch_size).to(device)
			optimizer.zero_grad()
			latent_mean, latent_logvar, mean, disp, pi = model(x_norm)
			KLD_loss = KLD(latent_mean,latent_logvar)
			zinb_loss = model.zinb_loss(x_raw,mean, disp, pi,batch_sf)
			loss = KLD_loss + zinb_loss
			KLDsave += KLD_loss.item()
			zinbsave += zinb_loss.item()
			overall_loss += loss.item()

			loss.backward()
			optimizer.step()

		#record loss
		record = [epoch,epochs,(overall_loss / (batch_idx * batch_size)),latent_size,lr,layers_data,KLDsave,zinbsave,batch_size]
		f = open(outFile, 'a')
		s = ','.join(str(x) for x in record)
		f.write(s)
		f.write('\n')
		f.close()


	return overall_loss

#datafiles = ['/Users/graha880/Desktop/mbVAE_resources/full_geographic_train_data.csv','/Users/graha880/Desktop/mbVAE_resources/subsampled_geographic_train_data.csv']

#datafiles = ["/Users/graha880/Desktop/mbVAE_resources/50_geographic_train_data.csv","/Users/graha880/Desktop/mbVAE_resources/100_geographic_train_data.csv","/Users/graha880/Desktop/mbVAE_resources/300_geographic_train_data.csv","/Users/graha880/Desktop/mbVAE_resources/600_geographic_train_data.csv"]

datafiles = ['/Users/graha880/Desktop/mbVAE_resources/full_geographic_train_data.csv']
layers_options = [[1000,700,400,100],[1000,500,300,100],[1000,800,600,400,200,100]]
#layers_options = [[1000,700,400,100]]
###batch_size_options = [100,50,20]
#lr_options = [1e-4,1e-5]
lr_options = [1e-5]
#epochs = [10,30,50]
epochs = [20]
latent_sizes = [10,25]
batch_sizes = [50,100]
#latent_sizes = [10]
#results = []
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
	outFile = '/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/' + type + '_tuning_hyperparms_' + date + '.csv'
	f = open(outFile,'w')
	header=['curr_epoch','epochs','loss','latent_dim','learning_rate','layers','kld_loss','zinb_loss','batch_size']
	s = ','.join(str(x) for x in header)
	f.write(s)
	f.write('\n')
	f.close()
	start_time = time.perf_counter()
	for layers in layers_options:
		print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
		print(layers)
		for batch_size in batch_sizes:
			train_loader = DataLoader(dataset=combined_training_set, batch_size=batch_size, shuffle=True,
									  drop_last=True)
			for lr in lr_options:
				for epoch in epochs:
					for latent in latent_sizes:
						model = VAE(layers_data=layers.copy(),input_dim=1422,latent_dim=latent).to(device) #need to move input dim to end of args but i'm too lazy rn
						optimizer = torch.optim.Adam(model.parameters(),lr=lr)
						layersStr = [str(i) for i in layers]
						layersStr = "-".join(layersStr)
						total_loss = tune_hyperparameters(model,lr,optimizer,epochs=epoch,layers_data=layersStr,latent_size=latent,outFile=outFile,device=device,features=raw_train_dataset.shape[1],batch_size=batch_size,training_data=train_loader)
						#latent_mean, latent_logvar, mean, disp, pi = model(raw_train_dataset)
						#mse = torch.nn.functional.mse_loss(mean,raw_train_dataset)
						#iter_results = [layers, lr, epoch, latent, mse.detach().numpy().item(),total_loss]
						#results.append(iter_results)
		#results_save = results
	end_time = time.perf_counter()
	lapsed = end_time - start_time
	print(lapsed)
