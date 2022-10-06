import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils import data
import re
import random
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

class Config:
        seed = '5'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epochs = 200
        batch_size = 128
        hidden_dim = 256
        embed_dim = 256
        dropout = 0.1
        dec_max_len = 30
        MAX_LENGTH = 20
        teacher_forcing_ratio = 0.5
        n_layers = 2
        lr = 0.001
        graphemes = ['<pad>', '<unk>', '</s>', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'ä', 'è', 'é', 'ê', 'ë', 'í', 'ï', 'ó', 'ô', 'ö', 'ú', 'û']
        phonemes = ['<pad>', '<unk>', '<s>', '</s>', '2:', '9', '9y', '@', '@i', '@u', 'A:', 'E', 'N', 'O', 'Of', 'S', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h_', 'i', 'i@', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'u', 'u@', 'v', 'w', 'x', 'y', 'z', '{']   
        graph2index = {g: idx for idx, g in enumerate(graphemes)}
        index2graph = {idx: g for idx, g in enumerate(graphemes)}

        phone2index = {p: idx for idx, p in enumerate(phonemes)}
        index2phone = {idx: p for idx, p in enumerate(phonemes)}
        g_vocab_size = len(graphemes)
        p_vocab_size = len(phonemes)
cfg = Config()



class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, g_vocab_size, n_layers, dropout):
        super().__init__()
        self.embed = embed_dim
        self.hidden = hidden_dim
        self.embed = nn.Embedding(g_vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first = True)
    
    def forward(self, graph_seq, graph_seq_len):
        embed_inputs = self.embed(graph_seq)
        inputs = self.dropout(embed_inputs)

        #https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html
        #packs padded sequences into tensor
        input_tensor = pack_padded_sequence(inputs, graph_seq_len, batch_first=True, enforce_sorted=False)
        output, (hidden, context) = self.lstm(input_tensor)

        return hidden, context

class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, p_vocab_size, n_layers, dropout):
        super().__init__()
        self.embed = embed_dim
        self.hidden = hidden_dim
        self.embed = nn.Embedding(p_vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout) 
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers, batch_first = True)
        self.fc = nn.Linear(hidden_dim ,p_vocab_size) #Predicts output
    
    def forward(self, decoder_inputs, hidden_init, context_init):

        
        embed_inputs = self.embed(decoder_inputs)
        inputs = self.dropout(embed_inputs)

        #is already a tensor

        output, (hidden, context) = self.lstm(inputs, (hidden_init, context_init))


        #Scaling output
        activation_output = self.fc(output)
        
        
        return activation_output, hidden,context

class G2PModel(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.enc = encoder
        self.dec = decoder
        self.device = device
        
    def forward(self, graph_seq, graph_seq_len,phone_seq_len, decoder_inputs, phoneme_target_vec = None, training = False, teacher_forcing = cfg.teacher_forcing_ratio):
        
        #Obtain hidden and context vectors from encoder
        hidden_init, context_init = self.enc(graph_seq, graph_seq_len)
        hidden, context = hidden_init, context_init

        max_len = max(phone_seq_len)

        phoneme_input_vec = decoder_inputs[:, :1]
        outputs = [] 
        phone_pred_seq = []
            
        if training:
            for i in range(0, max_len):

                output, hidden, context = self.dec(phoneme_input_vec ,hidden, context)
                outputs.append(output)
                # phone_pred = torch.tensor(output.argmax(-1))

                if random.random() > teacher_forcing: 
                    phoneme_input_vec = phoneme_target_vec[:,i]
                    
                else:  phoneme_input_vec = decoder_inputs[:,i]
                phoneme_input_vec = torch.unsqueeze(phoneme_input_vec,1)

        else: #for prediction
            for i in range(1, cfg.dec_max_len+1):
                output, hidden, context = self.dec(phoneme_input_vec ,hidden, context)
                
                phone_pred = output.argmax(-1)
                outputs.append(output)
                phone_pred_seq.append(phone_pred)
                phoneme_input_vec = phone_pred
                #print(i)
                #print(phoneme_input_vec.shape)
            phone_pred_seq = torch.cat(phone_pred_seq, 1)
            


        output = torch.cat(outputs, 1)
        
        return output, phone_pred_seq

def cfg_init():
    class Config:
        seed = '5'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        epochs = 200
        batch_size = 128
        hidden_dim = 256
        embed_dim = 256
        dropout = 0.1
        dec_max_len = 30
        MAX_LENGTH = 20
        teacher_forcing_ratio = 0.5
        n_layers = 2
        lr = 0.001
        graphemes = ['<pad>', '<unk>', '</s>', "'", 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'á', 'ä', 'è', 'é', 'ê', 'ë', 'í', 'ï', 'ó', 'ô', 'ö', 'ú', 'û']
        phonemes = ['<pad>', '<unk>', '<s>', '</s>', '2:', '9', '9y', '@', '@i', '@u', 'A:', 'E', 'N', 'O', 'Of', 'S', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h_', 'i', 'i@', 'j', 'k', 'l', 'm', 'n', 'p', 'r', 's', 't', 'u', 'u@', 'v', 'w', 'x', 'y', 'z', '{']   
        graph2index = {g: idx for idx, g in enumerate(graphemes)}
        index2graph = {idx: g for idx, g in enumerate(graphemes)}

        phone2index = {p: idx for idx, p in enumerate(phonemes)}
        index2phone = {idx: p for idx, p in enumerate(phonemes)}
        g_vocab_size = len(graphemes)
        p_vocab_size = len(phonemes)
    
    cfg = Config()
    return cfg

        
def g2p_model_init():
    cfg = cfg_init()
    encoder = Encoder(cfg.embed_dim, cfg.hidden_dim, cfg.g_vocab_size, cfg.n_layers, cfg.dropout)
    decoder = Decoder(cfg.embed_dim, cfg.hidden_dim, cfg.g_vocab_size, cfg.n_layers, cfg.dropout)
    print(cfg.device)

    model = G2PModel(encoder, decoder, cfg.device)
    return model


    