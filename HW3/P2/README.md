In order to run this model, you must import ctcdecoder first via the following commands in the Jupyter notebook.

`!git clone --recursive https://github.com/parlance/ctcdecode.git`

`!pip install wget`

`%cd ctcdecode`

`!pip install .`

`%cd ..`


Then you can simply enter python3 hw3p2.py into the command line, and the script will train an RNN and test it on the validation data every epoch and save the model
at each epoch as a .pth file & will output a submission.csv file on the test data. For this model I used an LSTM structured recurrent neural network. 
In order to create a 'blank' symbol to use during CTC decoding, I adjusted the labels of the data by an increment of 1, to allow the model to use the 0th index of 
the label data as the 'blank' for phoneme translation. The model architecture I used was one with 4 bidirectional LSTM layers, with a dropout rate of 0.3, of size 512, with 2 Linear output 
layers of size 256 and 42, the latter of which was the number of potential phoneme classes, including the 'blank' symbol, followed by log-softmax activation. 
The model itself was trained with a batch size of 64, using Connectionist Temporal Classification Loss as the criterion. The model was optimized during training 
using the Adam optimizer, with a learning rate of 1e-3 & a weight_decay rate of 5e-6. Validation Loss was assessed by using the CTC Beam decoder, with a beam width of 20, 
and outputting the phoneme labels with the best bath of the 20 candidate paths, & performance on the validation set was assessed via Levenshtein distance as a metric. During training, 
the model was able to quickly reduce the Average Levenshtein distance between the output & target labels on the validation set to between 15-20 within the first 6 epochs. 
The model eventually converged at around 8 Levenshtein distance by the 24th epoch. I used a ReduceLROnPlateau scheduler, with a patience of 0 and minimum learning rate of 1e-4


