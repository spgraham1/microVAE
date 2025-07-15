# Data Processing
import pandas as pd
import torch
import numpy as np

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from vae import VAE

#this dataset contains CRC samples with diagnoses ranging from 0 (normal) to 4 (cancer), with 3 intermediate stages
#first we will attempt training a RF to predict all of the diagnoses

data = pd.read_csv('/Users/graha880/Desktop/mbVAE_resources/predictions/diabetes_counts.csv',sep=',')
meta = pd.read_csv('/Users/graha880/Desktop/mbVAE_resources/predictions/diabetes_metadata.csv',sep=',')
combined = pd.concat([meta,data],axis=1)



all_classes_mapping = {
	'NG': 0,
	"IFG": 1,
	"IGT": 2,
	"IFG_IGT": 3,
	"new_T2D": 4,
	"treated_previous_T2D": 5
}

diabetes_vs_all = {
	'NG': 0,
	"IFG": 0,
	"IGT": 0,
	"IFG_IGT": 0,
	"new_T2D": 1,
	"treated_previous_T2D": 1
}

normal_vs_all = {
	'NG': 0,
	"IFG": 1,
	"IGT": 1,
	"IFG_IGT": 1,
	"new_T2D": 1,
	"treated_previous_T2D": 1
}

collapsed_categories ={
	'NG': 0,
	"IFG": 1,
	"IGT": 1,
	"IFG_IGT": 1,
	"new_T2D": 2,
	"treated_previous_T2D": 2
}
mapping_strategies = [diabetes_vs_all, normal_vs_all, collapsed_categories, all_classes_mapping]


LAYERS = [1000, 800, 600, 400, 200, 100] #based on hyperparameter tuning
LATENT = 10
device = 'cpu'
models = ['/Users/graha880/Desktop/mbVAE_resources/trained_models/full_model.pt', '/Users/graha880/Desktop/mbVAE_resources/trained_models/subsampled_model.pt']

RF_accuracies = []
MLP_accuracies = []
embeddedRF_accuracies = []
embeddedMLP_accuracies = []
maps = []
modelType = []
classSize = []
for infile in models:
	pretrainedModel = VAE(layers_data=LAYERS.copy(), input_dim=1422, latent_dim=LATENT).to(device)
	pretrainedModel.load_state_dict(torch.load('/Users/graha880/Desktop/mbVAE_resources/trained_models/full_model.pt'))
	currModel = infile.replace('/Users/graha880/Desktop/mbVAE_resources/trained_models/','')
	currModel = currModel.replace('_model.pt','')
	print(currModel)
	#dataNP = data.to_numpy()
	for myMap in mapping_strategies:
		print(myMap)
		#the classes are imbalanced, so we need to downsample some of the classes
		combined['status_encoded'] = combined['value'].map(myMap) # encode the status by the map
		minCount = min(combined['status_encoded'].value_counts()) #find the minimum class size
		downsampled = pd.DataFrame(columns=combined.columns) #create an empty df to store the downsampled data

	   # iterate through the classes and select minCount random samples from each class
		# then add the selected samples to the downsampled df
		for i in pd.unique(combined['status_encoded']):
			temp = combined[combined['status_encoded'] == i]
			temp2 = resample(temp,replace=False,n_samples=minCount)
			if (i == pd.unique(combined['status_encoded'])[0]):
				downsampled = temp2
			else:
				downsampled = pd.concat([downsampled,temp2])
		for i in range(100):
			classSize.append(minCount)
			maps.append(myMap)
			modelType.append(currModel)

			dataNP = downsampled[data.columns].to_numpy()
			print("Class balance: \n", downsampled.status_encoded.value_counts())
			metaNP = downsampled['status_encoded'].values
			x_train, x_test, y_train, y_test = train_test_split(dataNP,metaNP,test_size=0.2)
			rf = RandomForestClassifier(random_state=12)
			rf.fit(x_train,y_train)
			y_pred = rf.predict(x_test)
			RF_accuracy = accuracy_score(y_test, y_pred)
			RF_accuracies.append(RF_accuracy)
			print("RF Accuracy:", RF_accuracy)

			mlp = MLPClassifier(random_state=12,hidden_layer_sizes=(800, 200), activation='relu', solver='adam',max_iter=5000)
			mlp.fit(x_train, y_train)
			mlp_y_pred = mlp.predict(x_test)


			mlp_accuracy = accuracy_score(y_test, mlp_y_pred)
			MLP_accuracies.append(mlp_accuracy)
			print('MLP Accuracy: ', mlp_accuracy)


			x_train_tensor = torch.tensor(x_train,dtype=torch.float32)
			x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

			embedded_x_train = pretrainedModel.encode(x_train_tensor)[0].detach().numpy()
			embedded_x_test = pretrainedModel.encode(x_test_tensor)[0].detach().numpy()

			embeddedMLP = MLPClassifier(random_state=12,hidden_layer_sizes=(10,5),activation='relu',solver='adam',max_iter=5000)
			embeddedMLP.fit(embedded_x_train,y_train)
			mlp_embedded_y_pred = embeddedMLP.predict(embedded_x_test)
			mlp_embedded_accuracy = accuracy_score(y_test,mlp_embedded_y_pred)
			embeddedMLP_accuracies.append(mlp_embedded_accuracy)
			print('Embedded MLP Accuracy: ',mlp_embedded_accuracy)

			embeddedRF = RandomForestClassifier(random_state=12)
			embeddedRF.fit(embedded_x_train, y_train)
			RF_embedded_y_pred = embeddedMLP.predict(embedded_x_test)
			RF_embedded_accuracy = accuracy_score(y_test, RF_embedded_y_pred)
			embeddedRF_accuracies.append(RF_embedded_accuracy)
			print('Embedded RF Accuracy: ', RF_embedded_accuracy)


accuracy_results = {'RF': RF_accuracies, 'MLP': MLP_accuracies, 'embedded_RF': embeddedRF_accuracies, 'embedded_MLP': embeddedMLP_accuracies, 'mapping_strategy': maps, 'class_size': classSize, 'model': modelType}
accuracy_results = pd.DataFrame(accuracy_results)
accuracy_results.to_csv('/Users/graha880/mbVAE_resources/predictions/diabetes_prediction_accuracies.csv')