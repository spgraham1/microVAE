import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.preprocessing import StandardScaler


#read in the data that can be used for geography test prediction later on
test_dataset = pd.read_csv('~/Desktop/mbVAE_resources/geographic_test_data.csv',sep=',')
test_dataset = test_dataset.to_numpy()
test_dataset = torch.tensor(test_dataset,dtype=torch.float32)

train_dataset = pd.read_csv('~/Desktop/mbVAE_resources/geographic_train_data.csv',sep=',')
sample_counts = train_dataset.sum(axis=1)
size_factor = sample_counts / np.median(sample_counts)
raw = train_dataset.to_numpy() # get the raw training data
scaler = StandardScaler()
norm = scaler.fit_transform(train_dataset)

# convert all to tensors
raw_train_dataset = torch.tensor(raw,dtype=torch.float32)
norm_train_dataset = torch.tensor(norm,dtype=torch.float32)
size_factor = torch.tensor(size_factor,dtype=torch.float32)

combined_training_set = TensorDataset(norm_train_dataset, raw_train_dataset, size_factor)

def load_training_data(infile):
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

	return(combined_training_set)


#read the training data
#train_dataset = pd.read_csv('~/Desktop/mbVAE/training_data_100k.tsv',sep='\t')
#train_dataset = train_dataset.to_numpy()
#train_dataset = torch.tensor(train_dataset,dtype=torch.float32)
#test_dataset = pd.read_csv('~/Desktop/mbVAE/test_data.tsv',sep='\t')
#test_dataset = pd.read_csv('~/Desktop/mbVAE_resources/balanced_test_set.tsv',sep='\t')


#train_dataset = pd.read_csv('~/Desktop/mbVAE/raw_balanced_samples.csv',sep=',')

#norm = ad.AnnData(raw) # copy the raw dataset to normalize
#sc.pp.normalize_total(norm) # get the normalized dataset


