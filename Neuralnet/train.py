import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
import os
import torchvision
from torchvision import datasets, transforms
from pytorch_lightning.metrics.functional import accuracy

AVAIL_GPUS = min(1, torch.cuda.device_count())
BATCH_SIZE = 256 if AVAIL_GPUS else 64

class EncoderCNN(pl.LightningModule):
  def __init__(self, embed_size = 1024):
    super(EncoderCNN, self).__init__()
    
    # get the pretrained densenet model
    self.densenet = torchvision.models.densenet121(pretrained=True)
    # replace the classifier with a fully connected embedding layer
    self.densenet.classifier = nn.Linear(in_features=1024, out_features=1024)
    # add another fully connected layer
    self.embed = nn.Linear(in_features=1024, out_features=embed_size)
    # dropout layer
    self.dropout = nn.Dropout(p=0.5)
    # activation layers
    self.prelu = nn.PReLU()
    
  def forward(self, images):
    # get the embeddings from the densenet
    densenet_outputs = self.dropout(self.prelu(self.densenet(images)))
    # pass through the fully connected
    embeddings = self.embed(densenet_outputs)
    
    return embeddings

class DecoderRNN(pl.LightningModule):
  def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
    super(DecoderRNN, self).__init__()

    # define the properties
    self.embed_size = embed_size
    self.hidden_size = hidden_size
    self.vocab_size = vocab_size

    # lstm cell
    self.lstm_cell = nn.LSTMCell(input_size=embed_size, hidden_size=hidden_size)
    # output fully connected layer
    self.fc_out = nn.Linear(in_features=self.hidden_size, out_features=self.vocab_size)
    # embedding layer
    self.embed = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embed_size)
    # activations
    self.softmax = nn.Softmax(dim=1)
  
  def forward(self, features, captions):      
    # batch size
    batch_size = features.size(0)
    
    # init the hidden and cell states to zeros
    hidden_state = torch.zeros((batch_size, self.hidden_size)).cuda()
    cell_state = torch.zeros((batch_size, self.hidden_size)).cuda()
    # define the output tensor placeholder
    outputs = torch.empty((batch_size, captions.size(1), self.vocab_size)).cuda()
    # embed the captions
    captions_embed = self.embed(captions)
    
    # pass the caption word by word
    for t in range(captions.size(1)):
      # for the first time step the input is the feature vector
      if t == 0:
        hidden_state, cell_state = self.lstm_cell(features, (hidden_state, cell_state))
      # for the 2nd+ time step, using teacher forcer
      else:
        hidden_state, cell_state = self.lstm_cell(captions_embed[:, t, :], (hidden_state, cell_state))
      # output of the attention mechanism
      out = self.fc_out(hidden_state)
      # build the output tensor
      outputs[:, t, :] = out

    return outputs

   
net = EncoderCNN()
decoder = DecoderRNN()