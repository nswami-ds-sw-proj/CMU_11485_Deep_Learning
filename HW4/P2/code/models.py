import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.autograd import Variable
import util
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import random
class Attention(nn.Module):
    '''
    Attention is calculated using key, value and query from Encoder and decoder.
    Below are the set of operations you need to perform for computing attention:
        energy = bmm(key, query)
        attention = softmax(energy)
        context = bmm(attention, value)
    '''
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, query, key, value, lens):
        '''
        :param query :(batch_size, hidden_size) Query is the output of LSTMCell from Decoder
        :param keys: (batch_size, max_len, encoder_size) Key Projection from Encoder
        :param values: (batch_size, max_len, encoder_size) Value Projection from Encoder
        :return context: (batch_size, encoder_size) Attended Context
        :return attention_mask: (batch_size, max_len) Attention mask that can be plotted 
        '''
        query = query.unsqueeze(2)
        key = key.permute(1,0,2)
        energy = torch.bmm(key, query).squeeze(2)
        mask = torch.arange(key.shape[1]).unsqueeze(0).to(DEVICE) >= lens.unsqueeze(1).float().to(DEVICE)
        energy.masked_fill_(mask, 1e-9)
        attention = torch.nn.functional.softmax(energy, dim=1)

        value = value.permute(1,2,0)
        new_attn = attention.unsqueeze(2)
        context = torch.bmm(value, new_attn)
        context = context.squeeze(2)
        return context, attention
class pBLSTM(nn.Module):
    '''
    Pyramidal BiLSTM
    The length of utterance (speech input) can be hundereds to thousands of frames long.
    The Paper reports that a direct LSTM implementation as Encoder resulted in slow convergence,
    and inferior results even after extensive training.
    The major reason is inability of AttendAndSpell operation to extract relevant information
    from a large number of input steps.
    '''
    def __init__(self, input_dim, hidden_dim):
        super(pBLSTM, self).__init__()
        self.blstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        self.dropout = LockedDropout()
    def forward(self, x):
        '''
        :param x :(N, T) input to the pBLSTM
        :return output: (N, T, H) encoded sequence from pyramidal Bi-LSTM 
        '''
        x, lens = utils.rnn.pad_packed_sequence(x)
        x = self.dropout(x, 0.2)
        half_len = x.size(0) // 2
        feats = x.size(2) * 2
        ## NEED TO VET THIS THROUGH
        # print(x.shape)
        x_new = x[:(2*half_len), :, :]
        # print(x_new.shape)
        x_new = x_new.permute(1,0,2)
        # print(x_new.shape)
        x_new = x_new.reshape(x.shape[1], half_len, feats)
        # print(x_new.shape)
        x_new = x_new.permute(1,0,2)
        # print(x_new.shape)
        lens //= 2 

        x_new = utils.rnn.pack_padded_sequence(x_new, lens, enforce_sorted=False)
        output, lens = self.blstm(x_new)
        return output, lens

class Encoder(nn.Module):
    '''
    Encoder takes the utterances as inputs and returns the key and value.
    Key and value are nothing but simple projections of the output from pBLSTM network.
    '''
    def __init__(self, input_dim, hidden_dim, value_size=128,key_size=128):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, bidirectional=True)
        
        ### Add code to define the blocks of pBLSTMs! ###
        self.pBLSTM1 = pBLSTM(hidden_dim*4, hidden_dim)
        self.pBLSTM2 = pBLSTM(hidden_dim*4, hidden_dim)
        self.pBLSTM3 = pBLSTM(hidden_dim*4, hidden_dim)
        # NEED TO ADD DROPOUT
        self.dropout = LockedDropout()
        self.key_network = nn.Linear(hidden_dim*2, value_size)
        self.value_network = nn.Linear(hidden_dim*2, key_size)

    def forward(self, x, lens):
        rnn_inp = utils.rnn.pack_padded_sequence(x, lengths=lens, batch_first=False, enforce_sorted=False)
        outputs, _ = self.lstm(rnn_inp)

        ### Use the outputs and pass it through the pBLSTM blocks! ###
        outputs, lens = self.pBLSTM1(outputs)
        outputs, lens = self.pBLSTM2(outputs)
        outputs, lens = self.pBLSTM3(outputs)
        # NEED TO ADD DROPOUT
        linear_input, lens = utils.rnn.pad_packed_sequence(outputs)
        linear_input = self.dropout(linear_input, 0.2)
        keys = self.key_network(linear_input)
        value = self.value_network(linear_input)

        return keys, value, lens


class Decoder(nn.Module):
    '''
    As mentioned in a previous recitation, each forward call of decoder deals with just one time step, 
    thus we use LSTMCell instead of LSLTM here.
    The output from the second LSTMCell can be used as query here for attention module.
    In place of value that we get from the attention, this can be replace by context we get from the attention.
    Methods like Gumble noise and teacher forcing can also be incorporated for improving the performance.
    '''
    def __init__(self, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=True):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.lstm1 = nn.LSTMCell(input_size=hidden_dim + value_size, hidden_size=512)
        self.lstm2 = nn.LSTMCell(input_size=512, hidden_size=key_size)
        self.value_size = value_size
        self.isAttended = isAttended
        if (isAttended == True):
            self.attention = Attention()

        self.character_prob = nn.Linear(key_size + value_size, vocab_size)
        self.character_prob.weight = self.embedding.weight
    def forward(self, key, values, input_lengths, epoch=1, batch_num=None, plot=False, text=None, isTrain=True):
        '''
        :param key :(T, N, key_size) Output of the Encoder Key projection layer
        :param values: (T, N, value_size) Output of the Encoder Value projection layer
        :param text: (N, text_len) Batch input of text with text_length
        :param isTrain: Train or eval mode
        :return predictions: Returns the character perdiction probability 
        '''
        batch_size = key.shape[1]

        if (isTrain == True):
            max_len =  text.shape[1]
            embeddings = self.embedding(text)
        else:
            max_len = 600

        predictions = []
        hidden_states = [None, None]
        prediction = (torch.ones(batch_size,1)*33).long().to(DEVICE)#(torch.ones(batch_size, 1)*33).to(DEVICE)
        attentions = []
        context = values[0,:,:]
        tf = 0.9
        if epoch >= 10:
            tf = 0.85
        elif epoch >= 18 and epoch <= 24:
            tf = 0.8
        elif epoch >= 25 and epoch <=28: 
            tf = 0.75
        elif epoch >= 29 and epoch <=33:
            tf = 0.7
        elif epoch >= 34 and epoch <=38:
            tf = 0.65
        elif epoch >= 39:
            tf = 0.6
        if not self.training:
            tf = 1
        #CHANGE UP LATER
        print("TF = %f" % tf)
        for i in range(max_len):
            # * Implement Gumble noise and teacher forcing techniques 
            # * When attention is True, replace values[i,:,:] with the context you get from attention.
            # * If you haven't implemented attention yet, then you may want to check the index and break 
            #   out of the loop so you do not get index out of range errors. 
            
            if i==0:
                char_embed  = self.embedding(prediction)
                char_embed = char_embed.squeeze(1)
            else:
                if (isTrain) and torch.rand(1) < tf:
                    char_embed = embeddings[:,i-1,:].to(DEVICE)

                else:
                    prediction = torch.nn.functional.softmax(prediction, dim=1)
                    char_embed = self.embedding(prediction.argmax(dim=-1))
                    char_embed = char_embed.squeeze(1)

            inp = torch.cat([char_embed, context], dim=1)
            hidden_states[0] = self.lstm1(inp, hidden_states[0])

            inp_2 = hidden_states[0][0]
            hidden_states[1] = self.lstm2(inp_2, hidden_states[1])

            ### Compute attention from the output of the second LSTM Cell ###
           
            output = hidden_states[1][0]
            assert(self.isAttended)
            context, attention = self.attention(output, key, values, input_lengths)
            attentions.append(attention[0, :])
            prediction = self.character_prob(torch.cat([output, context], dim=1))

            predictions.append(prediction.unsqueeze(1))
        if plot and isTrain and len(attentions) > 1:
            assert(batch_num is not None)
            attentions = torch.stack(attentions, dim=0).detach().to("cpu")
            util.plot_attn_flow(attentions, 'attention_%d_%d.jpeg' % (epoch, batch_num))

        return torch.cat(predictions, dim=1)


class Seq2Seq(nn.Module):
    '''
    We train an end-to-end sequence to sequence model comprising of Encoder and Decoder.
    This is simply a wrapper "model" for your encoder and decoder.
    '''
    def __init__(self, input_dim, vocab_size, hidden_dim, value_size=128, key_size=128, isAttended=True):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim)
        self.decoder = Decoder(vocab_size, hidden_dim)

    def forward(self, speech_input, text_input, speech_len, epoch, batch_num=None, plot=False, isTrain=True):
        key, value, lens = self.encoder(speech_input, speech_len)
        if (isTrain == True):
            predictions = self.decoder(key, value, lens, epoch=epoch, batch_num=batch_num, plot=plot,text=text_input)
        else:
            predictions = self.decoder(key, value, lens, epoch=epoch, batch_num=batch_num, plot=plot,isTrain=False)
        return predictions
class LockedDropout(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dropout=0.5):
        if not self.training or not dropout:
            return x
        m = x.data.new(1, x.size(1), x.size(2)).bernoulli_(1 - dropout)
        mask = Variable(m, requires_grad=False) / (1 - dropout)
        mask = mask.expand_as(x)
        return mask * x

