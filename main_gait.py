from HMMUtil import HMMUtil
from gait_datareader import GaitDataReader
from adaboost_ensembles import AdaboostEnsembles
import numpy as np
# read dataset
gait_reader = GaitDataReader()
gait_reader.read()
# fit hmms
hmm_util_raw_data = HMMUtil()
# hmm_util_raw_data.fit(gait_reader.train_data,num_gaussians=[2]*5,num_states=3,trans_mat=np.array([[0.5, 0.5, 0.0],[0.0, 0.5, 0.5],[0.0, 0.0, 1.0]]))
hmm_util_raw_data.fit(gait_reader.train_data,num_gaussians=[4]*5,num_states=4,n_iterations=5,start_prob=np.array([1.0,0.0,0.0,0.0])
                      ,trans_mat=np.array([[0.5,0.5,0.0,0.0],[0.0,0.5,0.5,0.0],[0.0,0.0,0.5,0.5],[0.0,0.0,0.0,1.0]]))

# construct ensembles
ensembles = AdaboostEnsembles()
ensembles.fit(hmm_util_raw_data.mapping_observation_state[:,:-1], hmm_util_raw_data.mapping_observation_state[:,-1])
# train HMMs on new features got from the ensembles
new_train_data = []
for frames in gait_reader.train_data:
    new_train_data_inner_list = []
    for frame in frames:
        new_train_data_inner_list.append(ensembles.ensemble_scores(frame))
    new_train_data.append(new_train_data_inner_list)

hmm_ensemble_feats = HMMUtil()
hmm_ensemble_feats.fit(new_train_data,num_gaussians=[2]*5,num_states=3,trans_mat=np.array([[0.5, 0.5, 0.0],[0.0, 0.5, 0.5],[0.0, 0.0, 1.0]])
                       ,start_prob= np.array([1.0,0.0,0.0]))

for class_index,frames in enumerate(gait_reader.test_data):
    new_test_inner_list = []
    for frame in frames:
        new_test_inner_list.append(ensembles.ensemble_scores(frame))
    #store predictions
    pred_class_indices = hmm_ensemble_feats.predict(new_test_inner_list)
    a = np.argmax(pred_class_indices,axis=1)













