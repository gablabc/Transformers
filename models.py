import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import math, copy
import numpy as np
from simple_parsing import ArgumentParser
from dataclasses import dataclass


# class Embeddings(nn.Module):
#     def __init__(self, self.hparams.d_embed, vocab):
#         super(Embeddings, self).__init__()
#         self.lut = nn.Embedding(vocab, self.hparams.d_embed)
#         self.self.hparams.d_embed = self.hparams.d_embed

#     def forward(self, x):
#         return self.lut(x) * math.sqrt(self.self.hparams.d_embed)


class Embeddings(nn.Module):
    def __init__(self, d_embed, ntokens):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(ntokens, d_embed)
        self.d_embed = d_embed
        self.ntokens = ntokens

    def forward(self, x):
        x = torch.round((self.ntokens - 1) * x.squeeze(2)).type(torch.long)
        return self.lut(x) * math.sqrt(self.d_embed)
    


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_embed, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_embed)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_embed, 2) *
                             -(math.log(10000.0) / d_embed))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # To apply to the whole data batch
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)



def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0



def attention(query, key, value, mask=None, dropout=None):
    """
    Compute the attention mechanisms which maps an sequence of
    embeddings to another sequence of embedings

    Parameters
    ----------
    query : (..., N, d') Tensor

    key : (..., N, d') Tensor

    value : (..., N, d') Tensor

    Returns
    -------
    embedding : (..., N, d') Tensor

    attention_map : (..., N, N) Tensor
    """
    d_prime = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_prime)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention_map = scores.softmax(dim=-1)
    if dropout is not None:
        attention_map = dropout(attention_map)
    embeddings = torch.matmul(attention_map, value)
    return embeddings, attention_map



class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_embed, dropout=0.1):
        """ Take in model size and number of heads. """
        super(MultiHeadedAttention, self).__init__()
        assert d_embed % h == 0
        # We assume d_v always equals d_k
        self.d_prime = d_embed // h
        self.h = h
        self.linears = clones(nn.Linear(d_embed, d_embed), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, query, key, value, mask=None):
        """ 
        Forward pass, the three inputs are projected on lower dimension spaces
        d_prime, then passed through an attention mechanisms and the results 
        are concatenated 

        Parameters
        ----------
        query : (..., N, d') Tensor

        key : (..., N, d') Tensor

        value : (..., N, d') Tensor

        Returns
        -------
        embedding : (..., N, d') Tensor

        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from self.hparams.d_embed => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_prime).transpose(1, 2)
            for lin, x in zip(self.linears[:-1], (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_prime)
        )
        del query
        del key
        del value
        embedding = self.linears[-1](x)
        return embedding



class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std  = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_embed, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_embed, d_hid)
        self.w_2 = nn.Linear(d_hid, d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, d_embed, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    



class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_embed, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_embed, dropout), 2)
        self.d_embed = d_embed

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)



class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_embed)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    



class TransformerModel(nn.Module):

    @dataclass
    class HParams():
        d_embed: int = 20    # Dimension of the embedding
        n_head: int = 4      # Number of attention heads
        n_layers: int = 4    # Number of layers
        n_hid: int = 40      # Number of hidden units in feedforward components
        n_tokens: int = 3    # Number of tokens
        n_class: int = 10    #  Number of classes to predict
        dropout: int = 0.1  # Dropout probability
        device: str = "Default"


    def __init__(self, hparams: HParams=None, **kwargs):
        """
        Initialization of the Ensemble class. The user can either give a premade 
        hparams object made from the Hparams class or give the keyword arguments 
        to make one.

        Args:
            hparams (HParams, optional): HParams object to specify the models 
            characteristics. Defaults to None.
        """
        self.hparams= hparams or self.HParams(**kwargs)  
        super().__init__()

        # Unless device is specified, use GPUs if available
        if self.hparams.device == "Default":
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.hparams.device)
        self.name = "Transformer"

        # Initialize the layers objects
        self.mask = subsequent_mask(784).to(self.device)
        self.embedder = Embeddings(self.hparams.d_embed, self.hparams.n_tokens)
        self.pos_encoder = PositionalEncoding(self.hparams.d_embed, self.hparams.dropout)
        self.transformer_encoder = Encoder(
                                            EncoderLayer(
                                                self.hparams.d_embed, 
                                                MultiHeadedAttention(self.hparams.n_head, self.hparams.d_embed), 
                                                PositionwiseFeedForward(self.hparams.d_embed, self.hparams.n_hid), 
                                                dropout=self.hparams.dropout
                                                ), 
                                            self.hparams.n_layers
                                        )
        self.decoder = nn.Linear(self.hparams.d_embed, self.hparams.n_class)
        self.loss_fun = nn.CrossEntropyLoss()

        # Initialize the parameter values
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, X):
        E = self.embedder(X)
        E = self.pos_encoder(E)
        output = self.transformer_encoder(E, self.mask)
        output = self.decoder(output[:, -1, :])
        return output
    
    # Generic loss function
    def loss(self, X, y):
        # Fast forward pass on GPU
        X = X.to(self.device)
        y_pred = self(X)
        y = y.to(self.device)
        
        loss = self.loss_fun(y_pred, y)
        
        # # Regularisation
        # if self.hparams.regul_lambda:
        #     weights = self.all_params.split(self.predictor.split_sizes, dim = 1)[::2]
        #     for weight in weights: 
        #         loss += self.hparams.regul_lambda * torch.norm(weight, 'fro') ** 2
        
        return loss
    

    @classmethod
    def from_argparse_args(cls, args):
        """Creates an instance of the Class from the parsed arguments."""
        hparams: cls.HParams = args.model
        return cls(hparams)


    @classmethod
    def add_argparse_args(cls, parser: ArgumentParser):
        """Adds command-line arguments for this Class to an argument parser."""
        parser.add_arguments(cls.HParams, "model")
