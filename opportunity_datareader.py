from scipy.stats import stats
from format_data import DataReader
import Constants
from util import sliding_window
from feature_extractor import calculate_features
import numpy as np
import pickle

class OpportunityDataReader:

    def read(self):

        self.train_data = []
        self.test_data = []
        # self test data is a list of tuples. each tuple corresponds to a single file
        # each tuple contains a list of numpy arrays and contains a numpy array containing sample by sample
        # predictions
        self.validation_data = []
        self.class_list = [
            (0,      'Other'),
            (406516, 'Open Door 1'),
            (406517, 'Open Door 2'),
            (404516, 'Close Door 1'),
            (404517, 'Close Door 2'),
            (406520, 'Open Fridge'),
            (404520, 'Close Fridge'),
            (406505, 'Open Dishwasher'),
            (404505, 'Close Dishwasher'),
            (406519, 'Open Drawer 1'),
            (404519, 'Close Drawer 1'),
            (406511, 'Open Drawer 2'),
            (404511, 'Close Drawer 2'),
            (406508, 'Open Drawer 3'),
            (404508, 'Close Drawer 3'),
            (408512, 'Clean Table'),
            (407521, 'Drink from Cup'),
            (405506, 'Toggle Switch')
        ]

        data_reader = DataReader("/home/vishvak/LargeScaleFeatureLearning/OpportunityUCIDataset/dataset")
        train_raw_data_frames = data_reader.data["training"]
        # populating training data sequences
        # iterating through training files
        for data_frame in train_raw_data_frames:
            train_labels = data_frame[:, -1]
            # removing the label column
            train_raw_data = data_frame[:, :-1]
            mask = train_labels != 0
            train_labels = train_labels[mask]
            train_raw_data = train_raw_data[mask,:]
            # here we have removed the null data
            # we need to get the sequences corresponding to the other classes
            # and append them to our data list
            class_labels = np.unique(train_labels)
            if len(self.train_data) == 0:
                # initialize the list
                for _ in range(class_labels.shape[0]):
                    self.train_data.append([])
            indices_dict = self.get_indices(train_labels)
            for i,class_label in enumerate(class_labels):
                train_data_inner_list = []
                # add numpy array corresponding to sequences to this list
                for index in indices_dict[class_label]:
                    raw_data = train_raw_data[index[0]:index[1],:]
                    # we need to convert this into feature vector sequences
                    feats,_ = sliding_window(raw_data,train_labels[index[0]:index[1]],Constants.sliding_window_size,
                                                       Constants.overlap, calculate_features)
                    if np.all(feats !=  None):
                        train_data_inner_list.append(feats)
                    else:
                        print('small window' + 'class label : {}'.format(class_label))

                self.train_data[i] = self.train_data[i] + train_data_inner_list

        test_raw_data_frames = data_reader.data["test"]
        # populating testing data sequences
        # iterating through test files
        for data_frame in test_raw_data_frames:
            test_labels = data_frame[:, -1]
            # removing the label column
            test_raw_data = data_frame[:, :-1]
            # class_labels = np.unique(test_labels)
            # for i,class_label in enumerate(class_labels):
            #     self.class_list[class_label] = i

            # if len(self.test_data) == 0:
            #     # initialize the list
            #     for _ in range(class_labels.shape[0]):
            #         self.test_data.append([])

            # add numpy array corresponding to sequences to this list
            test_inner_list = []
            r = 0
            while (r + Constants.testing_frame_size)  < test_raw_data.shape[0]:
                raw_data = test_raw_data[r:r+Constants.testing_frame_size,:]
                raw_labels = test_labels[r:r+Constants.testing_frame_size]
                feats,_ = sliding_window(raw_data,raw_labels,Constants.testing_sliding_window_size,
                                                       Constants.testing_overlap, calculate_features)
                test_inner_list.append(feats)
                r += Constants.testing_frame_size
            '''@TODO we are missing the last few samples. these generally woudnt matter but think about 
            better ways of doing it '''
            self.test_data.append((test_inner_list,test_labels))
        train_data_file = open('trainFile','wb')
        test_data_file = open('testFile','wb')
        pickle.dump(self.train_data,train_data_file)
        pickle.dump(self.test_data,test_data_file)
        train_data_file.close()
        test_data_file.close()

    """
    returns the indices of the contiguous sequences in the input array
    labels : (n_samples,)
    
    returns: dictinary of list of tuples (start,stop) ; stop being inclusive. start and stop 
             correspond to one sequence. dictionary index with class label
    """
    def get_indices(self,labels):
        ret_dict = {}
        start = 0
        for i in range(labels.shape[0]):
            if labels[start] != labels[i]:
                if labels[start] in ret_dict:
                    ret_dict[labels[start]].append((start,i-1))
                else:
                    ret_dict[labels[start]] = [(start,i-1)]
                start = i

        if labels[start] in ret_dict:
            ret_dict[labels[start]].append((start, labels.shape[0]-1))
        else:
            ret_dict[labels[start]] = [(start, labels.shape[0]-1)]

        return ret_dict

reader = OpportunityDataReader()
reader.read()
pass