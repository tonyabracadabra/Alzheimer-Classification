from sklearn import datasets
from sklearn.cross_validation import train_test_split
from scipy import io as sio
from tensorflow.python.framework import ops
from dfs2 import DeepFeatureSelectionNew
import numpy as np
from sklearn.datasets import make_classification
from sklearn.preprocessing import normalize
from sklearn.preprocessing import Imputer

def setRegion(region):
    if region == 'PFC':
        ourdata = data_PFC
    elif region == 'VC':
        ourdata = data_VC
    elif region == 'CR':
        ourdata = data_CR
    elif region == 'All':
        ourdata = data_All
        
    inputX = ourdata['X']
    # Inpute using median
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(inputX)
    inputX = imp.transform(inputX)
    
    inputY = ourdata['y'][0,:]
    
    return inputX, inputY

data_All = sio.loadmat("/Volumes/TONY/LabWW/Alzheimer/GSE44772/GSE44772_data.mat")

data_PFC = sio.loadmat("/Volumes/TONY/LabWW/Alzheimer/GSE44772/GSE44772_PFC.mat")
data_VC = sio.loadmat("/Volumes/TONY/LabWW/Alzheimer/GSE44772/GSE44772_VC.mat")
data_CR = sio.loadmat("/Volumes/TONY/LabWW/Alzheimer/GSE44772/GSE44772_CR.mat")

for region in ['All']:
    print("Start training for region: "+region)
    inputX, inputY = setRegion(region)
    weights = []
    for i in xrange(100):
    	# Resplit the data
    	X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.2, random_state=i)

        # Change number of epochs to control the training time
    	dfsMLP = DeepFeatureSelectionNew(X_train, X_test, y_train, y_test, n_input=1, hidden_dims=[100], dropout=[False], \
                                     learning_rate=0.01, lambda1=0.001, lambda2=1, alpha1=0.1, alpha2=0, activation='tanh', weight_init='uniform', \
                                     epochs=50, optimizer='Adam', print_step=20)

    	dfsMLP.train(batch_size=200)
    	print("Train finised for random state: " + str(i)) + " on region: " + region
    	weights.append(dfsMLP.selected_ws[0])

    # The generated weights will be in the weights folder
    np.save("./weights/weights_"+region, weights)