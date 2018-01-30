import numpy as np
import format_data
import Constants
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
directory = "/home/vishvak/LargeScaleFeatureLearning/OpportunityUCIDataset/dataset"
reader = format_data.DataReader(directory=directory)
for i,test_frame in enumerate(reader.data['test']):
    scores = pd.read_csv('predictions'+str(i+1)+'.out',header=None)
    scores = scores.values
    le = LabelEncoder()
    labels = test_frame[:,-1]
    labels = le.fit_transform(labels)
    # label_mapping = {}
    # label_list = np.unique(labels)
    # for j,label in enumerate(label_list):
    #     label_mapping[label] = j
    preds = np.argmax(scores,axis=1)
    repeated_preds= np.repeat(preds,Constants.testing_frame_slide_samples)
    final_preds = np.zeros(labels.shape[0])
    final_preds[int(Constants.testing_frame_size/2):
                int(Constants.testing_frame_size/2)+ repeated_preds.shape[0]] = repeated_preds

    # final_preds = np.dot(preds,np.ones(Constants.testing_frame_size).reshape(1,-1))
    # labels and final preds have slightly differnt lengths as we ignore the last few samples which dont fall into the window
    difference_len = labels.shape[0] - final_preds.shape[0]
    final_preds = np.concatenate((final_preds,np.zeros(difference_len)))
    null_mask = labels == 0
    final_preds = final_preds[~null_mask]
    labels = labels[~null_mask]
    labels = labels - 1
    print(np.sum(labels == final_preds)/labels.shape[0])
    print(confusion_matrix(y_true=labels,y_pred=final_preds))