



# Part 1

In this section, I implemented Multi-Layer Perceptron (MLP) linear layers, activations, loss functions and basic optimization techniques in my own version of Pytorch, including engineering an auto-differentiation engine similar to that used by Pytorch during backpropagation. 


# Part 2

**Kaggle Challenge**: https://www.kaggle.com/c/11-785-fall-20-homework-1-part-2

This challenge involved applying knowledge of feedforward neural networks in order to recognize speech, specifically classifying audio frequency data into classes of phonemes.
The audio recordings were provided as multiple utterances composed of a variable number of frames, with accompanying phoneme labels for each frame. The data comes from LibriSpeech corpus which
is derived from audiobooks that are part of the LibriVox project, and contains 1000 hours of speech sampled
at 16 kHz. 
## Data
* **train.npy**
* **train_labels.npy**
* **dev.npy**
* **dev_labels.npy**
* **test.npy**

Code is all contained in **model.py**
