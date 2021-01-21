import numpy as np
import torch
from torch.utils.data import Dataset 
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence 

'''
Loading all the numpy files containing the utterance information and text information
'''
def load_data():
    speech_train = np.load('train.npy', allow_pickle=True, encoding='bytes')
    speech_valid = np.load('dev.npy', allow_pickle=True, encoding='bytes')
    speech_test = np.load('test.npy', allow_pickle=True, encoding='bytes')

    transcript_train = np.load('./train_transcripts.npy', allow_pickle=True,encoding='bytes')
    transcript_valid = np.load('./dev_transcripts.npy', allow_pickle=True,encoding='bytes')

    return speech_train, speech_valid, speech_test, transcript_train, transcript_valid


'''
Transforms alphabetical input to numerical input, replace each letter by its corresponding 
index from letter_list
'''
def transform_letter_to_index(transcript, letter2index):
    '''
    :param transcript :(N, ) Transcripts are the text input
    :param letter_list: Letter list defined above
    :return letter_to_index_list: Returns a list for all the transcript sentence to index
    '''
    result = []
    
    for sentence in transcript:
        word_list = []
        for word in sentence:
            word = word.decode('utf-8')
            word_list.append(word)

        result.append(' '.join(word_list))
    char_labels = []
    for sentence in result:
        sentence_char_labels = []
        for char in sentence:
            sentence_char_labels.append(letter2index[char])
        sentence_char_labels.append(letter2index['<eos>'])
        # print(len(sentence_char_labels))
        # sentence_char_labels.insert(0, letter2index['<sos>'])
        char_labels.append(sentence_char_labels)
    return char_labels



'''
Optional, create dictionaries for letter2index and index2letter transformations
'''
def create_dictionaries(letter_list):
    letter2index = dict()
    index2letter = dict()
    for i in range(len(letter_list)):
        letter2index[letter_list[i]] = i
        index2letter[i] = letter_list[i]


    return letter2index, index2letter


class Speech2TextDataset(Dataset):
    '''
    Dataset class for the speech to text data, this may need some tweaking in the
    getitem method as your implementation in the collate function may be different from
    ours. 
    '''
    def __init__(self, speech, text=None, isTrain=True):
        self.speech = speech
        self.isTrain = isTrain
        if (text is not None):
            
            self.text = text

    def __len__(self):
        return self.speech.shape[0]

    def __getitem__(self, index):
        if (self.isTrain == True):
            return torch.tensor(self.speech[index].astype(np.float32)), torch.tensor(self.text[index])
        else:
            return torch.tensor(self.speech[index].astype(np.float32))


def collate_train(batch_data):
    ### Return the padded speech and text data, and the length of utterance and transcript ###
    speech, text = zip(*batch_data)
    speech_lens = []
    for audio in speech:
        speech_lens.append(len(audio))
    
    speech_lens = torch.LongTensor(speech_lens)
    pad_speech = pad_sequence(speech)
    pad_text = pad_sequence(text, batch_first=True) #ONLY PLACE TO BATCH FIRST
    return pad_speech, pad_text, speech_lens # Check on this

def collate_test(batch_data):
    ### Return padded speech and length of utterance ###
    speech = batch_data
    speech_lens = []
    for audio in speech:
        speech_lens.append(len(audio))
    
    speech_lens = torch.LongTensor(speech_lens)
    pad_speech = pad_sequence(speech)
    return pad_speech, speech_lens  #Check on this