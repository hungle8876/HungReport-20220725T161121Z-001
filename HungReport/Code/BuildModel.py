from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from joblib import dump, load
from scipy.fft import fft, fftfreq
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import librosa
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis, tstd, median_abs_deviation
SVM = 1
KNN = 2
RFR = 3
#
modelSelection = 1

#Parameter
nClasses = 6
nChunk = 24
fs = 250
T = 1/fs

def featureExtraction(signal):
    mfccs = librosa.feature.mfcc(signal, sr=250,n_mfcc=10)
    # print(mfccs.shape)
    mfccs_features = mfccs.reshape(mfccs.shape[0]*mfccs.shape[1])
    # amp_features = np.array([np.mean(signal), np.std(signal), median_abs_deviation(signal, scale='normal'), skew(signal), kurtosis(signal), np.sqrt(np.mean(signal**2)), np.min(signal), np.max(signal)])
    # print(amp_features.shape[0])
    # amp_features.reshape(1, amp_features.shape[0])
    # features = np.concatenate((mfccs_features,amp_features))
    features = mfccs_features
    return(features)

X_raw= np.zeros(shape=(nClasses*nChunk, 20))
X = np.zeros(shape=(nClasses*nChunk, 20))
Y = np.zeros(shape=(nClasses*nChunk, 1))

#Up
for i in range(nChunk):
    fileName = "Data/up/up_" + str(i+1) + ".csv"    
    x = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0) 
    feature1 =  featureExtraction(x[:,0]) 
    feature2 =  featureExtraction(x[:,1])  
    X_raw[i,:] = np.append(feature1,feature2)
Y[0:nChunk] = 1

#down
for i in range(nChunk):
    fileName = "Data/down/down_" + str(i+1) + ".csv"    
    x = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0) 
    feature1 =  featureExtraction(x[:,0]) 
    feature2 =  featureExtraction(x[:,1])  
    X_raw[nChunk+i,:] = np.append(feature1,feature2)
Y[nChunk:nChunk*2] = 2

#left
for i in range(nChunk):
    fileName = "Data/left/left_" + str(i+1) + ".csv"    
    x = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0) 
    feature1 =  featureExtraction(x[:,0]) 
    feature2 =  featureExtraction(x[:,1])  
    X_raw[2*nChunk+i,:] = np.append(feature1,feature2)
Y[2*nChunk:nChunk*3] = 3

#right
for i in range(nChunk):
    fileName = "Data/right/right_" + str(i+1) + ".csv"    
    x = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0) 
    feature1 =  featureExtraction(x[:,0]) 
    feature2 =  featureExtraction(x[:,1])  
    X_raw[3*nChunk+i,:] = np.append(feature1,feature2)
Y[3*nChunk:nChunk*4] = 4

#enter
for i in range(nChunk):
    fileName = "Data/enter/enter_" + str(i+1) + ".csv"    
    x = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0) 
    feature1 =  featureExtraction(x[:,0]) 
    feature2 =  featureExtraction(x[:,1])  
    X_raw[4*nChunk+i,:] = np.append(feature1,feature2)
Y[4*nChunk:nChunk*5] = 5

#no
for i in range(nChunk):
    fileName = "Data/no/no_" + str(i+1) + ".csv"    
    x = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0) 
    feature1 =  featureExtraction(x[:,0]) 
    feature2 =  featureExtraction(x[:,1])  
    X_raw[5*nChunk+i,:] = np.append(feature1,feature2)
Y[5*nChunk:nChunk*6] = 6




xTrain, xTest, yTrain, yTest = train_test_split( X_raw, Y, stratify=Y,train_size=0.8, test_size=0.2)
if modelSelection == SVM:
    model = LinearSVC(penalty = 'l2', C = .1380000000000001, tol = .008, max_iter = 1000, dual = False, class_weight = 'balanced')
elif modelSelection == KNN:
    model = KNeighborsClassifier(n_neighbors=3)
elif modelSelection == RFR:
    model = RandomForestClassifier(max_depth=2, random_state=0)
model.fit(xTrain, yTrain)
dump(model, "model.joblib")


#print("> Train set accuracy: {:.3f}".format(model.score(xTrain, yTrain)))
acc = model.score(xTest, yTest)
print("> Test set accuracy: {:.3f}".format(acc))
#print("> Entire dataset accuracy: {:.3f}".format(model.score(X, Y)))

yPred = model.predict(xTest)
precision, recall,  fscore, support = precision_recall_fscore_support(yTest, yPred)
print(precision)
print(recall)
print(fscore)

# scores = cross_val_score(model, X_raw, Y, cv=5)
# for x, acc in enumerate(scores):
#     print("Fold " + str(x+1) + ". Accuracy : " + "{:.2f}".format(acc))

plot_confusion_matrix(model, xTest, yTest)
if modelSelection == SVM:
    plt.title("Confusion matrix: SVM. Accuracy = " + "{:.2f}".format((acc)))
elif modelSelection == KNN:
    plt.title("Confusion matrix: KNN. Accuracy = " + "{:.2f}".format((acc)))
elif modelSelection == RFR:    
    plt.title("Confusion matrix: RFR. Accuracy = " + "{:.2f}".format((acc)))
plt.show()


