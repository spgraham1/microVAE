from vae import VAE, KLD
from load_data import combined_training_set, raw_train_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset
import datetime
import time

batch_size = 100
train_loader = DataLoader(dataset=combined_training_set, batch_size=batch_size, shuffle=True)
device='cpu'

def tune_hyperparameters(model,lr, optimizer, epochs, layers_data, latent_size, outFile, device,features):
	model.train() # set model to training mode
	for epoch in range(epochs):
		overall_loss = 0
		KLDsave = 0
		zinbsave = 0
		#for batch_idx, x in enumerate(train_loader):
		for batch_idx, (norm_data, raw_data, batch_sf) in enumerate(train_loader):
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
		record = [epoch,epochs,(overall_loss / (batch_idx * batch_size)),latent_size,lr,layers_data,KLDsave,zinbsave]
		f = open(outFile, 'a')
		s = ','.join(str(x) for x in record)
		f.write(s)
		f.write('\n')
		f.close()


	return overall_loss


layers_options = [[1000,700,400,100],[1000,500,300,100],[1000,800,600,400,200,100]]
#layers_options = [[1000,700,400,100]]
###batch_size_options = [100,50,20]
lr_options = [1e-4,1e-5]
#lr_options = [1e-5]
epochs = [10,30,50]
#epochs = [50]
latent_sizes = [10,25,50]
#latent_sizes = [10]
#results = []
date = datetime.date.today().strftime("%Y-%m-%d")
outFile = '/Users/graha880/Desktop/mbVAE_resources/training/tuning_hyperparms_' + date + '.csv'

f = open(outFile,'w')
header=['curr_epoch','epochs','loss','latent_dim','learning_rate','layers','kld_loss','zinb_loss']
s = ','.join(str(x) for x in header)
f.write(s)
f.write('\n')
f.close()
start_time = time.perf_counter()
for layers in layers_options:
	print(time.strftime("%Y-%m-%d %H:%M:%S",time.localtime()))
	print(layers)
	for lr in lr_options:
		for epoch in epochs:
			for latent in latent_sizes:
				model = VAE(layers_data=layers.copy(),input_dim=1422,latent_dim=latent).to(device) #need to move input dim to end of args but i'm too lazy rn
				optimizer = torch.optim.Adam(model.parameters(),lr=lr)
				layersStr = [str(i) for i in layers]
				layersStr = "-".join(layersStr)
				total_loss = tune_hyperparameters(model,lr,optimizer,epochs=epoch,layers_data=layersStr,latent_size=latent,outFile=outFile,device=device,features=raw_train_dataset.shape[1])
				#latent_mean, latent_logvar, mean, disp, pi = model(raw_train_dataset)
				#mse = torch.nn.functional.mse_loss(mean,raw_train_dataset)
				#iter_results = [layers, lr, epoch, latent, mse.detach().numpy().item(),total_loss]
				#results.append(iter_results)
#results_save = results
end_time = time.perf_counter()
lapsed = end_time - start_time
print(lapsed)
