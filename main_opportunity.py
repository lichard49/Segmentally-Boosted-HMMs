from HMMUtil import HMMUtil
from opportunity_datareader import OpportunityDataReader
from adaboost_ensembles import AdaboostEnsembles
from sklearn.metrics import f1_score
import numpy as np
import pickle

# read dataset

# opportunity_reader = OpportunityDataReader()
# opportunity_reader.read()
# datareader_filename = open('datareader','wb')
# pickle.dump(opportunity_reader,datareader_filename)
# datareader_filename.close()
datareader_filename = open('datareader','rb')
opportunity_reader =pickle.load(datareader_filename)
# fit hmms
hmm_util_raw_data = HMMUtil()
# we add a -1 as we are only fitting to non-null activities
hmm_util_raw_data.fit(opportunity_reader.train_data,num_gaussians=[8] * (len(opportunity_reader.class_list)-1)
                      ,num_states=4,n_iterations=5,start_prob=np.array([1.0,0.0,0.0,0.0])
                      ,trans_mat=np.array([[0.5,0.5,0.0,0.0],[0.0,0.5,0.5,0.0],[0.0,0.0,0.5,0.5],
                                          [0.0,0.0,0.0,1.0]]))
hmm_util_raw_filename = open('hmm_raw','wb')
pickle.dump(hmm_util_raw_data,hmm_util_raw_filename)
hmm_util_raw_filename.close()
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
hmm_ensemble_feats.fit(new_train_data,num_gaussians=[3] * (len(opportunity_reader.class_list)-1),num_states=3,
                       n_iterations=5,start_prob=np.array([1.0,0.0,0.0]),
                       trans_mat=np.array([[0.5, 0.5, 0.0],[0.0, 0.5, 0.5],[0.0, 0.0, 1.0]]))
hmm_ensemble_filename = open('hmm_ensemble','wb')
pickle.dump(hmm_ensemble_feats,hmm_ensemble_filename)
hmm_ensemble_filename.close()
# y_true = np.array([])
# y_pred = np.array([])
# iterating through file by file
filename_root = 'predictions'
file_number = 1
ensembles_file = open('ensemble_object','wb')
pickle.dump(ensembles, ensembles_file)
for frames, Y in opportunity_reader.test_data:
    new_test_inner_list= []
    for frame in frames:
         new_test_inner_list.append(ensembles.ensemble_scores(frame))
    # predictions
    preds = hmm_ensemble_feats.predict(new_test_inner_list)
    np.savetxt(filename_root + str(file_number) + '.out', preds,delimiter=',')
    file_number +=1

# for class_index,frames in enumerate(opportunity_reader.test_data):
#     new_test_inner_list = []
#     for frame in frames:
#         new_test_inner_list.append(ensembles.ensemble_scores(frame))
#     #store predictions
#     y_pred = np.concatenate((y_pred,hmm_ensemble_feats.predict(new_test_inner_list)))
#     y_true = np.concatenate((y_true,np.ones(len(frames))))
#

# print(f1_score(y_pred=y_pred,y_true=y_true,average='macro'))