from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM,GMMHMM
import numpy as np

class HMMUtil:
    # class which deals with initializing one HMM per class using Mixture of Gaussians
    def __init__(self):
        self.HMMList = []

    # data : list of lists. Inner list contains numpy arrays corresponding to a sequence. Length of outer list is the number of classes
    def fit(self, data, num_gaussians, num_states, n_iterations = 10,trans_mat = None,start_prob = None):
        #@TODO play with covariance types and compare results
        frame_0 = data[0][0]
        num_features = frame_0.shape[1]
        num_classes = len(data)
        self.mapping_observation_state = np.empty((0,num_features + 1)) # add a 1 to add a row for getting corresponding states

        # iterate through each class and train a HMM
        for class_index,frames in enumerate(data):
            # train this model
            X = np.concatenate(frames)
            lengths = np.zeros(len(frames),dtype=np.dtype(np.int32))
            for i,arr in enumerate(frames):
                # each arr is a numpy array and each row in this array represents an observation
                lengths[i] =arr.shape[0]
            scores = np.zeros(n_iterations)
            HMM_list = []
            for j in np.arange(n_iterations):
                lr = hmm.GMMHMM(n_components=num_states,n_mix=num_gaussians[class_index],init_params = "cm", params = "cmt")
                # lr.startprob_ = np.array([1.0, 0.0, 0.0])
                # lr.transmat_ = np.array([[0.5, 0.5, 0.0],[0.0, 0.5, 0.5],[0.0, 0.0, 1.0]])
                lr.transmat_ = trans_mat
                lr.startprob_ = start_prob
                lr.fit(X=X,lengths=lengths)

                # temp_HMM = GMMHMM(n_components=num_states, n_mix = num_gaussians[class_index])
                # temp_HMM.startprob_ = [1.0,0.0,0.0]
                # temp_HMM.transmat_ = trans_mat
                # temp_HMM.fit(X=X,lengths=lengths)
                scores[j] = lr.score(X,lengths=lengths)
                HMM_list.append(lr)
            # finding the HMM with the best fit
            cur_HMM = HMM_list[np.argmax(scores)]
            self.HMMList.append(cur_HMM)
            temp = cur_HMM.decode(X,lengths)
            states = temp[1]
            # we do this Hack to make sure that the state numbers are different for different classes
            states += num_states * class_index
            a = np.concatenate((X,states.reshape(-1,1)),axis=1)
            self.mapping_observation_state = np.concatenate((self.mapping_observation_state,a))

    # data : list of numpy arrays ,each numpy array represents a sequences
    def predict(self,data):
        scores = np.zeros((len(data),len(self.HMMList)))
        for i,arr in enumerate(data):
            scores[i,:] = self.get_likelihoods(arr)
        # return np.argmax(scores,axis=1)
        return scores

    def get_likelihoods(self,X):
        likelihoods = np.zeros(len(self.HMMList))
        for j,HMM_Model in enumerate(self.HMMList):
            likelihoods[j] = HMM_Model.score(X)
        return likelihoods
