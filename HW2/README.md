# CNNs and Facial Recognition / Verification


## Part 1
In this section, I implemented CNN libraries into my working implementation of MyTorch.
## Part 2
**Kaggle Challenge**: https://www.kaggle.com/c/11785-fall2020-hw2p2-makeup/leaderboard (Rank 2)

Given categorized facial images of various human subjects, the model would have to perform N-way classification to classify the image as depicting one of the N possible 
human subjects present in the image dataset. The dataset was processed using Pytorch's torchvision library, and the metric for scoring on Kaggle was the AUC score (see reference Link),
on the pairs of images in the test dataset. 

### Data

* classification_data
  * train_data
  * val_data
  * test_data

* verification_data
  * verification_pairs_test.txt










Reference Link:
https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=AUC%20represents%20the%20probability%20that,has%20an%20AUC%20of%201.0.



