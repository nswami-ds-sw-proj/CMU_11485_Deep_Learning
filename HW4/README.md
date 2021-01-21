# Language Models using RNNs | Attention Mechanisms and Memory Networks

## Part 1

In this part of the assignment, I trained a neural network that utilized RNNs to model and generate language at the word level. I trained this model on 
the WikiText-2 Language Moldeling Dataset. The training data is structured into articles of variable word length. As part of the pre-processing,
the text data is concatenated and split into sequences of length 50 words (as a hyperparameter). The model itself consists of an embedding layer of size 512,
3 LSTM layers of size 1024, and a linear layer the size of the vocabulary of the dataset. 


## Part 2

**Kaggle Challenge**: https://www.kaggle.com/c/makeup-11785-hw4p2-fall2020


In this Kaggle challenge, I implemented a seq2seq model, utilizing Attention memory networks, to translate speech data into transcription of text, at the character-level, using an Encoder and Decoder
network. End-to-end, the system would be able to transcribe a given speech utterance to its corresponding transcript, dividing the task into three sections (Listen, Attend, and Spell).
To understand pBLSTM, Attention:
https://arxiv.org/pdf/1508.01211.pdf


## Data
* train.npy
* dev.npy
* test.npy
* train_transcript.npy
* dev_transcript.npy





