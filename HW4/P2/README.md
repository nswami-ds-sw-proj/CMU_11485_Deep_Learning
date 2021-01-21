

The algorithm written and subsequent model produced by the code in this repository is based
on the Listen, Attend and Spell model. The model takes in speech data, and feeds it into an encoder,
which converts it to a latent represenation. This latent represenation is fed into a decoder model,
which outputs its corresponding text translation. The encoder employs an initial bidirectional LSTM layer (size 256) as well 
as 3 pyramidal bidirectional LSTM layers (size 1024) as well. In order to improve its performance, the Locked Dropout
regularization technique for recurrent networks, developed by the Salesforce Repo, was used for the encoder (p=0.2). 
The latent represenation output by the encoder was fed into an attention model,
which output a context for the decoder to intake as well as the previous character prediction. The decoder used an embedding layer (size 256),
followed by 2 LSTM Cells of sizes 512 and 128, respectively. The final output of the LSTM Cells would pass through a linear layer with its size equal to the length
of the vocabulary. As an additional regularization technique, weight tying was used between the embedding layer and linear layer. This model
was able to reach an average Levenshtein distance of 20.5 on the test dataset. The model optimized on its CrossEntropyLoss, using the Adam optimizer and an lr of 1e-3.
The model trained for ~80 epochs. In order to run this model,
enter python3 main.py into the command line (assuming the proper datasets are in the directory).
