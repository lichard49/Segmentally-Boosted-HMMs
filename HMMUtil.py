from hmmlearn.hmm import GMMHMM
import numpy as np

class HMMUtil:
    # class which deals with initializing one HMM per class using Mixture of Gaussians
    def __init__(self):
        self.HMMList = []

    # data : list of lists. Inner list contains numpy arrays corresponding to a sequence
    def fit(self, data, num_gaussians, num_states):
        frame_0 = data[0]
        num_features = frame_0.shape[0]
        self.mapping_observation_state = np.empty((0,num_features + 1)) # add a 1 to add a row for getting corresponding states

        # iterate through each class and train a HMM
        for frames in data:
            cur_HMM = GMMHMM(n_components=num_states,n_mix=num_gaussians)
            self.HMMList.append(cur_HMM)
            # train this model
            X = np.concatenate(frames)
            lengths = np.zeros(len(frames))
            for i,arr in enumerate(frames):
                # each arr is a numpy array and each row in this array represents an observation
                lengths[i] = arr.shape[0]
            #TODO repeat this multiple time to get best model -- use score
            cur_HMM.fit(X=X,lengths=lengths)
            _,states = cur_HMM.decode(X,lengths)
            #TODO look at the value of states they should be different for different HMM's but highly doubt this
            self.mapping_observation_state = np.concatenate([self.mapping_observation_state,np.append(X,states,axis=1)])

    # data : list of numpy arrays , each numpy array represents a sequences
    def predict(self,data):
        scores = np.zeros((len(data),len(self.HMMList)))
        for i,arr in data:
            scores[i,:] = self.get_likelihoods(arr)
        return np.argmax(scores,axis=0)

    def get_likelihoods(self,X):
        likelihoods = np.zeros(len(self.HMMList))
        for j,HMM_Model in enumerate(self.HMMList):
            likelihoods[j] = HMM_Model.score(X)
