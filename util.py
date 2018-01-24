# sliding window for 1 second
import sys
import numpy as np
from scipy import stats
import math
import Constants
# data : nd array num_samples * num_features
# function handle : calls this function on all frames and concatenates the result in the output
# sliding_size : number of seconds in sliding window if index_time is False else the number of samples in window
# overlap " between 0 -100 ; percentage overlap between consecutive frames
# if index_time = True the first column must be the timestamp column and should have time in milli seconds

def sliding_window(data,labels,sliding_window_size,overlap,function_handle,index_time = False):

   if(sliding_window_size < 0 or overlap < 0 or overlap >100 or Constants.smaller_sliding_window> data.shape[0]):
       print("incorrect input formatting")
       return (None,None)
   if data.shape[0] > Constants.smaller_sliding_window and data.shape[0] < Constants.sliding_window_size:
       sliding_window_size = Constants.smaller_sliding_window

   # initialize the output array
   # detect size of output

   r,c = data.shape
   if r > 1000:
       pass
   print("Shape of input array:",data.shape)
   temp = function_handle(data[0:sliding_window_size,:])
   output = None
   output_labels = None
   # run the sliding window
   if not index_time:
       overlap_samples = int(np.floor(sliding_window_size * ((100.0 - overlap)/100.0)))
       print(overlap_samples)
       num_rows_output = int((r - sliding_window_size)/overlap_samples) + 1
       output = np.zeros((num_rows_output,temp.shape[0]))
       output_labels = np.zeros(num_rows_output)
       start =  0
       stop = sliding_window_size
       rindex = 0
       while stop < r:
           window = data[start:stop,:]
           output[rindex,:] = function_handle(window)
           m = stats.mode(labels[start:stop],axis=None)
           output_labels[rindex] = m[0]
           rindex += 1
           start = start + overlap_samples
           stop = start + sliding_window_size
   return output,output_labels


def sliding_window_v2(data,labels,sliding_window_size,overlap,function_handle,index_time = False):

   if(sliding_window_size < 0 or overlap < 0 or overlap >100 or sliding_window_size > data.shape[0]):
       print("incorrect input formatting")
       sys.exit(1)

   # initialize the output array
   # detect size of output

   r,c = data.shape
   print("Shape of input array:",data.shape)
   temp = function_handle(data[0:sliding_window_size,:])
   output = output_labels = None
   # run the sliding window
   if not index_time:
       print(temp.shape[0])
       print(int(np.floor(sliding_window_size * ((100.0 - overlap)/100.0))))
       output = np.zeros((int(np.ceil(r/int(np.floor(sliding_window_size * ((100.0 - overlap)/100.0))))) + 1 ,temp.shape[0]))
       output_labels = np.zeros((int(np.ceil(r/int(np.floor(sliding_window_size * ((100.0 - overlap)/100.0))))) + 1 ))
       start =  0
       stop = sliding_window_size - 1
       rindex = 0
       while start < r-10:
           label_window = labels[start:stop]
           unique_labels = np.unique(label_window)
           if len(unique_labels) != 1:
               # adjust start and stop
               #@TODO this could be dangerous
               new_label = label_window[-1]

               # if np.min(np.where(label_window == new_label)[0]) < int(np.floor(sliding_window_size * ((100.0 - overlap)/100.0))):
               #   start = start + np.min(np.where(label_window == new_label)[0])
               #   stop = start + sliding_window_size
               # else:
               #   stop = start + np.min(np.where(label_window == new_label)[0])

               start = start + np.min(np.where(label_window == new_label)[0])
               stop = start + sliding_window_size
               print("In If")
               print("start:",start)
               print("stop:",stop)

           window = data[start:stop, :]
           features = function_handle(window)
           output[rindex, :] = features
           output_labels[rindex] = label_window[0]
           start = start + int(np.floor(sliding_window_size * ((100.0 - overlap)/100.0)))
           stop = start + sliding_window_size
           rindex += 1
           print("start:", start)
           print("stop:", stop)

   mask = np.all(output == 0,axis=1)
   output_labels = output_labels[~mask]
   output = output[~mask,:]
   return output,output_labels




