import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models import Seq2Seq
from train_test import train, test, val
from dataloader import load_data, collate_train, collate_test, transform_letter_to_index, Speech2TextDataset, create_dictionaries

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LETTER_LIST = ['<pad>', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', \
               'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '-', "'", '.', '_', '+', ' ','<sos>','<eos>']

def main():
    model = Seq2Seq(input_dim=40, vocab_size=len(LETTER_LIST), hidden_dim=256, isAttended=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Check on this
    nepochs = 80
    batch_size = 64 if DEVICE == 'cuda' else 1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.95)
    letter2idx, index2letter = create_dictionaries(LETTER_LIST)
    speech_train, speech_valid, speech_test, transcript_train, transcript_valid = load_data()
    character_text_train = transform_letter_to_index(transcript_train, letter2idx)
    character_text_valid = transform_letter_to_index(transcript_valid, letter2idx)

    train_dataset = Speech2TextDataset(speech_train, character_text_train)
    val_dataset = Speech2TextDataset(speech_valid, character_text_valid)
    test_dataset = Speech2TextDataset(speech_test, None, False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_train)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_train)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_test)

    for epoch in range(nepochs):
        train(model, train_loader, criterion, optimizer, epoch)
        val(model, val_loader, epoch)
        scheduler.step()
        if epoch > 5 and epoch % 3 ==0:
            test(model, test_loader, epoch)


if __name__ == '__main__':
    main()