# Data Processing
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt

# Modeling
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_curve, auc, roc_auc_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from vae import VAE

#this dataset contains CRC samples with diagnoses ranging from 0 (normal) to 4 (cancer), with 3 intermediate stages
#first we will attempt training a RF to predict all of the diagnoses

data = pd.read_csv('/Users/graha880/Desktop/mbVAE_resources/predictions/CRC_counts.csv',sep=',')
meta = pd.read_csv('/Users/graha880/Desktop/mbVAE_resources/predictions/CRC_metadata.csv',sep=',')
combined = pd.concat([meta,data],axis=1)



cancer_vs_all = {
    'normal': 0,
    'high risk normal': 0,
    'adenoma': 0,
    'adv adenoma': 0,
    'cancer': 1
}



#dataNP = data.to_numpy()
#mapping_strategies = [cancer_vs_all,normal_vs_all,collapsed_categories,all_classes_mapping]


LAYERS = [1000, 800, 600, 400, 200, 100] #based on hyperparameter tuning
LATENT = 10
device = 'cpu'
#models = ['/Users/graha880/Desktop/mbVAE_resources/trained_models/full_model.pt', '/Users/graha880/Desktop/mbVAE_resources/trained_models/subsampled_model.pt']

RF_accuracies = []
MLP_accuracies = []
embeddedRF_accuracies = []
embeddedMLP_accuracies = []
maps = []
#modelType = []
classSize = []

RF_fpr = []
RF_tpr = []
RF_thresh = []

	#dataNP = data.to_numpy()
#for myMap in mapping_strategies:
myMap = cancer_vs_all
print(myMap)
#the classes are imbalanced, so we need to downsample some of the classes
combined['status_encoded'] = combined['value'].map(myMap) # encode the status by the map
minCount = min(combined['status_encoded'].value_counts()) #find the minimum class size
downsampled = pd.DataFrame(columns=combined.columns) #create an empty df to store the downsampled data

REPS = 100
roc_curve_data = pd.DataFrame(columns=['fpr','tpr','thresholds','auc','model','rep'])
roc_curve_data.to_csv('/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/predictions/CRC_ROC_data.csv')

# iterate through the classes and select minCount random samples from each class
# then add the selected samples to the downsampled df
for i in pd.unique(combined['status_encoded']):
	temp = combined[combined['status_encoded'] == i]
	temp2 = resample(temp,replace=False,n_samples=minCount)
	if (i == pd.unique(combined['status_encoded'])[0]):
		downsampled = temp2
	else:
		downsampled = pd.concat([downsampled,temp2])
for i in range(REPS):
	classSize.append(minCount)
	maps.append(myMap)
	#modelType.append(currModel)

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
	rf_pred_proba = rf.predict_proba(x_test)[:,1]  # get the prediction probability for class 1 (non-normal glucose)
	fpr, tpr, thresholds = roc_curve(y_test, rf_pred_proba)
	fpr, tpr, thresholds = roc_curve(y_test, rf_pred_proba)
	rf_roc_auc = auc(fpr, tpr)
	model_list = ["rf"] * fpr.size
	rep_list = [i] * fpr.size
	auc_list = [rf_roc_auc] * fpr.size
	rf_roc_df = pd.DataFrame(
		{'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': auc_list, 'model': model_list, 'rep': rep_list})

	rf_importances = rf.feature_importances_
	feature_importances_df = pd.DataFrame({'feature': data.columns, 'importance': rf_importances})
	feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
	feature_importances_df.to_csv('/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/predictions/CRC_RF_feature_importance.csv')
	if i == 0:
		feature_importances_df.to_csv('/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/predictions/CRC_RF_feature_importance.csv')
	else:
		feature_importances_df.to_csv('/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/predictions/CRC_RF_feature_importance.csv',mode='a',header=False)



	mlp = MLPClassifier(random_state=12,hidden_layer_sizes=(800, 200), activation='relu', solver='adam',max_iter=5000)
	mlp.fit(x_train, y_train)
	mlp_y_pred = mlp.predict(x_test)
	mlp_accuracy = accuracy_score(y_test, mlp_y_pred)
	MLP_accuracies.append(mlp_accuracy)
	print('MLP Accuracy: ', mlp_accuracy)
	mlp_pred_proba = mlp.predict_proba(x_test)[:,1]  # get the prediction probability for class 1 (non-normal glucose)
	mlp_fpr, mlp_tpr, mlp_thresholds = roc_curve(y_test, mlp_pred_proba)
	mlp_roc_auc = auc(mlp_fpr, mlp_tpr)

	# add these items to a df to plot later
	model_list = ["mlp"] * mlp_fpr.size
	rep_list = [i] * mlp_fpr.size
	auc_list = [mlp_roc_auc] * mlp_fpr.size
	mlp_roc_df = pd.DataFrame({'fpr': mlp_fpr, 'tpr': mlp_tpr, 'thresholds': mlp_thresholds, 'auc': auc_list, 'model': model_list,'rep': rep_list})

	#for infile in models:
	pretrainedModel = VAE(layers_data=LAYERS.copy(), input_dim=1422, latent_dim=LATENT).to(device)
	infile = '/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/trained_models/full_model.pt'
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
	print('Embedded MLP Accuracy: ',mlp_embedded_accuracy)
	embeddedMLP_accuracies.append(mlp_embedded_accuracy)
	embedded_mlp_pred_proba = embeddedMLP.predict_proba(embedded_x_test)[:, 1]  # get the prediction probability for class 1 (non-normal glucose)
	emlp_fpr, emlp_tpr, emlp_thresholds = roc_curve(y_test, embedded_mlp_pred_proba)
	emlp_roc_auc = auc(emlp_fpr,emlp_tpr)

	#add these items to a df to plot later
	model_list = ["embeddedMLP"] * emlp_fpr.size
	rep_list = [i] * emlp_fpr.size
	auc_list = [emlp_roc_auc] * emlp_fpr.size
	emlp_roc_df = pd.DataFrame({'fpr': emlp_fpr, 'tpr': emlp_tpr, 'thresholds': emlp_thresholds, 'auc': auc_list, 'model': model_list, 'rep': rep_list})


	embeddedRF = RandomForestClassifier(random_state=12)
	embeddedRF.fit(embedded_x_train, y_train)
	RF_embedded_y_pred = embeddedRF.predict(embedded_x_test)
	RF_embedded_accuracy = accuracy_score(y_test, RF_embedded_y_pred)
	embeddedRF_accuracies.append(RF_embedded_accuracy)
	print('Embedded RF Accuracy: ', RF_embedded_accuracy)
	embedded_rf_pred_proba = embeddedRF.predict_proba(embedded_x_test)[:,
							  1]  # get the prediction probability for class 1 (non-normal glucose)
	eRF_fpr, eRF_tpr, eRF_thresholds = roc_curve(y_test, embedded_rf_pred_proba)
	eRF_roc_auc = auc(eRF_fpr, eRF_tpr)
	eRF_importances = embeddedRF.feature_importances_

	#add these items to a df to plot later
	model_list = ["embeddedRF"] * eRF_fpr.size
	rep_list = [i] * eRF_fpr.size
	auc_list = [eRF_roc_auc] * eRF_fpr.size
	erf_roc_df = pd.DataFrame({'fpr': eRF_fpr, 'tpr':  eRF_tpr, 'thresholds':  eRF_thresholds, 'auc': auc_list, 'model': model_list, 'rep': rep_list})

	dfs = [rf_roc_df, mlp_roc_df, emlp_roc_df, erf_roc_df]
	merged = pd.concat(dfs)
	merged.to_csv('/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/predictions/CRC_ROC_data.csv', mode='a', header=False)

	feature_importances_df = pd.DataFrame({'feature': range(10), 'importance': eRF_importances})
	feature_importances_df = feature_importances_df.sort_values(by='importance', ascending=False)
	feature_importances_df.to_csv('/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/predictions/CRC_eRF_feature_importance.csv')

rf_pred_proba = rf.predict_proba(x_test)[:,1] #get the prediction probability for class 1 (non-normal glucose)
fpr, tpr, thresholds = roc_curve(y_test,rf_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"AUC: {roc_auc:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=1, label=f'RF (area = {roc_auc:.2f})')
plt.plot(mlp_fpr, mlp_tpr, color='green', lw=1, label=f'MLP (area = {mlp_roc_auc:.2f})')
plt.plot(emlp_fpr, emlp_tpr, color='pink', lw=1, label=f'Embedded MLP (area = {emlp_roc_auc:.2f})')
plt.plot(eRF_fpr, eRF_tpr, color='orange', lw=1, label=f'Embedded RF (area = {eRF_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve for Colorectal Cancer Status')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

accuracy_results = {'RF': RF_accuracies, 'MLP': MLP_accuracies, 'embedded_RF': embeddedRF_accuracies, 'embedded_MLP': embeddedMLP_accuracies, 'mapping_strategy': maps, 'class_size': classSize}
accuracy_results = pd.DataFrame(accuracy_results)
accuracy_results.to_csv('/Users/graha880/Desktop/mbVAE_resources/full_vs_subsample/softplus_models/predictions/CRC_prediction_accuracies.csv')