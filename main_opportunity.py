from HMMUtil import HMMUtil
from opportunity_datareader import OpportunityDataReader
from adaboost_ensembles import AdaboostEnsembles
from sklearn.metrics import f1_score
import numpy as np
# read dataset
opportunity_reader = OpportunityDataReader()
opportunity_reader.read()
# fit hmms
hmm_util_raw_data = HMMUtil()
# we add a -1 as we are only fitting to non-null activities
hmm_util_raw_data.fit(opportunity_reader.train_data,num_gaussians=[3] * (len(opportunity_reader.class_list)-1),num_states=5,n_iterations=5)
# construct ensembles
ensembles = AdaboostEnsembles()
ensembles.fit(hmm_util_raw_data.mapping_observation_state[:,:-1], hmm_util_raw_data.mapping_observation_state[:,-1])
# train HMMs on new features got from the ensembles
new_train_data = []
for frames in opportunity_reader.train_data:
    new_train_data_inner_list = []
    for frame in frames:
        new_train_data_inner_list.append(ensembles.ensemble_scores(frame))
    new_train_data.append(new_train_data_inner_list)

hmm_ensemble_feats = HMMUtil()
hmm_ensemble_feats.fit(new_train_data,num_gaussians=[2] * (len(opportunity_reader.class_list)-1),num_states=3)

y_true = np.array([])
y_pred = np.array([])
# iterating through file by file
for frames, Y in opportunity_reader.test_data:
    new_test_inner_list= []
    for frame in frames:
         new_test_inner_list.append(ensembles.ensemble_scores(frame))
    # predictions
    preds = hmm_ensemble_feats.predict(new_test_inner_list)

# for class_index,frames in enumerate(opportunity_reader.test_data):
#     new_test_inner_list = []
#     for frame in frames:
#         new_test_inner_list.append(ensembles.ensemble_scores(frame))
#     #store predictions
#     y_pred = np.concatenate((y_pred,hmm_ensemble_feats.predict(new_test_inner_list)))
#     y_true = np.concatenate((y_true,np.ones(len(frames))))

print(f1_score(y_pred=y_pred,y_true=y_true,average='macro'))