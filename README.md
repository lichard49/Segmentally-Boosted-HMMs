# Segmentally Boosted HMMs

An implementation of the segmentally boosted hidden Markov models as described in [Pei Yin's disseration](https://smartech.gatech.edu/handle/1853/33939).

Currently in a sandbox state.  I tried to use a [HTK](http://htk.eng.cam.ac.uk/) workspace as the starting point, but getting the required HMM information out of HTK turned out to be trickier than expected, so I also started from scratch by implementing the Viterbi algorithm and adding Adaboost to it (my_viterbi.py).

Included two datasets to start playing with: acceleglove, using accelerometers on a glove to recognize American sign language (ASL); and starner97, a dataset of videos also for recognizing ASL.
