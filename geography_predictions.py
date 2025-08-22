# Data Processing
import pandas as pd
import torch
import numpy as np

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from vae import VAE

#this dataset contains CRC samples with diagnoses ranging from 0 (normal) to 4 (cancer), with 3 intermediate stages
#first we will attempt training a RF to predict all of the diagnoses

#data = pd.read_csv('/Users/graha880/Desktop/mbVAE_resources/geographic_test_data.csv',sep=',')
#meta = pd.read_csv('/Users/graha880/Desktop/mbVAE_resources/geographic_test_metadata.csv',sep=',')
combined = pd.read_csv('/Users/graha880/Desktop/mbVAE_resources/geographic_test_metadata.csv',sep=',')
metadata_columns = ['sample','srs','project','srr','library_strategy','library_source',	'pubdate',	'total_bases',	'instrument',	'geo_loc_name',	'iso',	'region']
data_columns = list(set(combined.columns) - set(metadata_columns))
categories ={
	'Australia/New Zealand': 0,
	"Central and Southern Asia": 1,
	"Eastern and South-Eastern Asia": 2,
	"Europe and Northern America": 3,
	"Latin America and the Caribbean": 4,
	"Northern Africa and Western Asia": 5,
	"Sub-Saharan Africa": 6
}

LAYERS = [1000, 800, 600, 400, 200, 100] #based on hyperparameter tuning
LATENT = 10
device = 'cpu'
models = ['/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/trained_models/full_model.pt',
		  '/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/trained_models/subsampled_model.pt']

RF_accuracies = []
MLP_accuracies = []
embeddedRF_accuracies_full = []
embeddedMLP_accuracies_full = []
embeddedRF_accuracies_subsample = []
embeddedMLP_accuracies_subsample = []
maps = []
#modelType = []
classSize = []

mapping_strategies = [categories]
REPEAT = 100
for myMap in mapping_strategies:
	print(myMap)
	#the classes are imbalanced, so we need to downsample some of the classes
	combined['status_encoded'] = combined['region'].map(myMap) # encode the status by the map
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
	for i in range(REPEAT):
		classSize.append(minCount)
		maps.append(myMap)
		#modelType.append(currModel)

		dataNP = downsampled[data_columns].to_numpy()
		print("Class balance: \n", downsampled.status_encoded.value_counts())
		metaNP = downsampled['status_encoded'].values
		x_train, x_test, y_train, y_test = train_test_split(dataNP,metaNP,test_size=0.2)
		rf = RandomForestClassifier(random_state=12)
		rf.fit(x_train,y_train)
		y_pred = rf.predict(x_test)
		RF_accuracy = accuracy_score(y_test, y_pred)
		RF_accuracies.append(RF_accuracy)
		print("RF Accuracy:", RF_accuracy)
		rf_importances = rf.feature_importances_
		feature_importances_df = pd.DataFrame({'feature': data.columns, 'importance': rf_importances})
		feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
		feature_importances_df.to_csv('/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/predictions/geography_RF_feature_importance.csv')

		mlp = MLPClassifier(random_state=12,hidden_layer_sizes=(800, 200), activation='relu', solver='adam',max_iter=5000)
		mlp.fit(x_train, y_train)
		mlp_y_pred = mlp.predict(x_test)


		mlp_accuracy = accuracy_score(y_test, mlp_y_pred)
		MLP_accuracies.append(mlp_accuracy)
		print('MLP Accuracy: ', mlp_accuracy)

		for infile in models:
			pretrainedModel = VAE(layers_data=LAYERS.copy(), input_dim=1422, latent_dim=LATENT).to(device)
			pretrainedModel.load_state_dict(torch.load(infile))
			currModel = infile.replace('/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/trained_models/', '')
			currModel = currModel.replace('_model.pt', '')
			print(currModel)

			x_train_tensor = torch.tensor(x_train,dtype=torch.float32)
			x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

			embedded_x_train = pretrainedModel.encode(x_train_tensor)[0].detach().numpy()
			embedded_x_test = pretrainedModel.encode(x_test_tensor)[0].detach().numpy()

			embeddedMLP = MLPClassifier(random_state=12,hidden_layer_sizes=(10,5),activation='relu',solver='adam',max_iter=5000)
			embeddedMLP.fit(embedded_x_train,y_train)
			mlp_embedded_y_pred = embeddedMLP.predict(embedded_x_test)
			mlp_embedded_accuracy = accuracy_score(y_test,mlp_embedded_y_pred)
			if (currModel == 'subsampled'):
				embeddedMLP_accuracies_subsample.append(mlp_embedded_accuracy)
			elif (currModel == "full"):
				embeddedMLP_accuracies_full.append(mlp_embedded_accuracy)
			#embeddedMLP_accuracies.append(mlp_embedded_accuracy)
			print('Embedded MLP Accuracy: ',mlp_embedded_accuracy)

			embeddedRF = RandomForestClassifier(random_state=12)
			embeddedRF.fit(embedded_x_train, y_train)
			RF_embedded_y_pred = embeddedRF.predict(embedded_x_test)
			RF_embedded_accuracy = accuracy_score(y_test, RF_embedded_y_pred)
			if (currModel == 'subsampled'):
				embeddedRF_accuracies_subsample.append(RF_embedded_accuracy)
			elif (currModel == "full"):
				embeddedRF_accuracies_full.append(RF_embedded_accuracy)
			#embeddedRF_accuracies.append(RF_embedded_accuracy)
			print('Embedded RF Accuracy: ', RF_embedded_accuracy)
			eRF_importances = embeddedRF.feature_importances_
			feature_importances_df = pd.DataFrame({'feature': range(10), 'importance': eRF_importances})
			feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
			outfile = '/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/predictions/geography_eRF_'+currModel+'_feature_importance.csv'
			feature_importances_df.to_csv(outfile)

accuracy_results = {'RF': RF_accuracies, 'MLP': MLP_accuracies, 'embedded_RF_full': embeddedRF_accuracies_full,'embedded_RF_subsample': embeddedRF_accuracies_subsample, 'embedded_MLP_full': embeddedMLP_accuracies_full, 'embedded_MLP_subsample': embeddedMLP_accuracies_subsample,'mapping_strategy': maps, 'class_size': classSize}
accuracy_results = pd.DataFrame(accuracy_results)
accuracy_results.to_csv('/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/predictions/geography_prediction_accuracies.csv')