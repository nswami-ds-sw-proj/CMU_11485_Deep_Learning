import time
import torch
### Add Your Other Necessary Imports Here! ###
import random
import util
import Levenshtein
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LETTER_LIST = ['<pad>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    model.to(DEVICE)
    start = time.time()

    # 1) Iterate through your loader
    for batchnum, (speech, text, speech_lens) in enumerate(train_loader):
        # 2) Use torch.autograd.set_detect_anomaly(True) to get notices about gradient explosion
        optimizer.zero_grad()
        # 3) Set the inputs to the device.
        speech = speech.to(DEVICE)
        text = text.to(DEVICE)
        speech_lens = speech_lens.to(DEVICE)

        # 4) Pass your inputs, and length of speech into the model.
        if batchnum%100==0:
            outputs = model(speech, text, speech_lens, epoch, batch_num=batchnum, plot=True, isTrain=True)
        else:
            outputs = model(speech, text, speech_lens, epoch, batch_num=None, plot=False, isTrain=True)
        # outputs = outputs.permute(1,0,2) # To give batch_first
        # 7) Use the criterion to get the loss.
        outputs = outputs.permute(0, 2, 1)
        loss = criterion(outputs, text)
        
        # 8) Use the mask to calculate a masked loss. 

        # 9) Run the backward pass on the masked loss. 
        loss.backward()
        # 10) Use torch.nn.utils.clip_grad_norm(model.parameters(), 2)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        # 11) Take a step with your optimizer
        
        optimizer.step()
        # 12) Normalize the masked loss

        # 13) Optionally print the training loss after every N batches
        
        if (batchnum % 100)==0:
            print(batchnum, loss.item())
        #     util.plot_attn_flow(attention[:, -1].detach().cpu().numpy(), 'attn_epoch_%d_%d.jpeg' % (epoch, batchnum))
        # print(batchnum, loss.item())
        
    if (epoch > 20 and epoch % 5 == 0):
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, "criterion": criterion}
        torch.save(checkpoint, 'checkpoint_%d.pth' %(epoch))
        save_check = torch.load('checkpoint_%d.pth' % (epoch))
        assert('model' in save_check)
        assert('optimizer' in save_check)
        assert('epoch' in save_check)
        assert('criterion' in save_check)
        print("WORKS")
        

    end = time.time()

def val(model, val_loader, epoch):
    model.eval()

    model.to(DEVICE)
    with torch.no_grad():
        # output_pred = []
        # text_translation = []
        distances = []
        print(len(val_loader))
        for batch_num, (speech, text, speech_lens) in enumerate(val_loader):
            
            speech = speech.to(DEVICE)
            text = text.to(DEVICE)
            speech_lens = speech_lens.to(DEVICE)
            outputs = model(speech, text, speech_lens, epoch, plot=False,isTrain=True) # Check for TF in self.training()
            # outputs = outputs.permute(1,0,2)
            assert(outputs.shape[0]==1)
            assert(text.shape[0]==1)
            outputs = outputs.squeeze(0)
            text = text.squeeze(0)
            
            sentence1 = ''
            for char_dist in outputs:
                prediction = torch.argmax(char_dist)
                letter = LETTER_LIST[prediction]
                if (letter != '<pad>') and (letter != '<eos>'):
                    sentence1 += letter
                else:
                    break
            # output_pred.append(sentence1)
            
            sentence2 = ''
            for idx in text:
                letter = LETTER_LIST[idx.item()]
                if (letter != '<pad>') and (letter != '<eos>'):
                    sentence2 += letter
                else:
                    break
            # text_translation.append(sentence2)
            distances.append(Levenshtein.distance(sentence1, sentence2))
            # assert(len(output_pred) > batch_num)
            if batch_num<6:
                
                print("Model Output")
                print(sentence1)
                print("TEXT LABEL")
                print(sentence2)

            
        # assert(len(text_translation)==len(output_pred) and len(text_translation)==len(val_loader))

    print("AVG VAL DISTANCE = %d, EPOCH %d" % (np.mean(distances), epoch))

def test(model, test_loader, epoch):
    ### Write your test code here! ###
    model.eval()
    model.to(DEVICE)
    outfile = open('submission_%d.csv' % (epoch), 'w+')
    outfile.write('id,label\n')
    with torch.no_grad():
        for batch_num, (speech, speech_lens) in enumerate(test_loader):
            speech = speech.to(DEVICE)
            speech_lens = speech_lens.to(DEVICE)
            outputs = model(speech, None, speech_lens, epoch, plot=False, isTrain=False)
            # outputs = outputs.permute(1,0,2)
            assert(outputs.shape[0]==1)
            outputs = outputs.squeeze(0)
            sentence = ''
            for char_dist in outputs:
                prediction = torch.argmax(char_dist)
                letter = LETTER_LIST[prediction]
                if (letter != '<pad>') and (letter != '<eos>'):
                    sentence += letter
                else:
                    break
            outfile.write(str(batch_num) + ',' + sentence + '\n')
    outfile.close()