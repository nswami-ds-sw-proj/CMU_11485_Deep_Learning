# RNNs and Connectionist Temporal Classification


## Part 1
In this section, I implemented RNN libraries into my working implementation of MyTorch, including GRU and LSTM units.
## Part 2
**Kaggle Challenge**: https://www.kaggle.com/c/makeup-11785-hw3p2-fall2020/overview

Input data is an array of utterances, each of which is shaped (frames, frequencies). 
Labels are simply the list of phonemes in the utterance [0-40]. There are 41 phoneme labels. 
The phoneme array will be as long as however many phonemes are in the utterance. 
There is a lookup file mapping each phoneme to a single character for the purposes of this competition.
The task in this challenge was to classify each frame to a phoneme, using RNNs. The training data contains **unaligned** phonemes; alignment
was not required for this challenge.


### Data

* train.npy
* dev.npy
* test.npy
* train_labels.npy
* dev_labels.npy
* phoneme_list.py

