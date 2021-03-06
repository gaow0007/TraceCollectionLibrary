import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Dropout, LayerNorm, TransformerEncoder
from multiheadattention import  InProjContainer, ScaledDotProduct , MultiheadAttentionContainer

import adaptdl.torch 

import copy 
from typing import Tuple, Optional 


@adaptdl.torch.fx.wrap 
def enumerate_range(x): 
    S, N = x.size() 
    pos = torch.arange(S,
                        dtype=torch.long,
                        device=x.device).unsqueeze(0).expand((N, S)).t()
    return pos 

@adaptdl.torch.fx.wrap 
def zeroShape(S, N, seq_input, token_type_input): 
    if token_type_input is not None: 
        return token_type_input
    return torch.zeros((S, N), dtype=torch.long).to(seq_input.device)

@adaptdl.torch.fx.wrap
def split_input(inputs): 
    return torch.transpose(inputs[0], 0, 1), inputs[1]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x): 
        pos = enumerate_range(x)
        return self.pos_embedding(pos)



class TokenTypeEncoding(nn.Module):
    def __init__(self, type_token_num, d_model):
        super(TokenTypeEncoding, self).__init__()
        self.token_type_embeddings = nn.Embedding(type_token_num, d_model)

    def forward(self, seq_input, token_type_input):
        S, N = seq_input.size() 
        token_type_input = zeroShape(S, N, seq_input, token_type_input) 

        return self.token_type_embeddings(token_type_input)


class BertEmbedding(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5):
        super(BertEmbedding, self).__init__()
        self.ninp = ninp
        self.ntoken = ntoken
        self.pos_embed = PositionalEncoding(ninp)
        self.embed = nn.Embedding(ntoken, ninp)
        self.tok_type_embed = TokenTypeEncoding(2, ninp)  # Two sentence type
        self.norm = LayerNorm(ninp)
        # self.dropout = Dropout(dropout)

    def forward(self, src, token_type_input): 
        src = self.embed(src) + self.pos_embed(src) \
            + self.tok_type_embed(src, token_type_input)
        # return self.dropout(self.norm(src))
        return self.norm(src)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="gelu"):
        super(TransformerEncoderLayer, self).__init__()
        in_proj_container = InProjContainer(Linear(d_model, d_model),
                                            Linear(d_model, d_model),
                                            Linear(d_model, d_model))
        self.mha = MultiheadAttentionContainer(nhead, in_proj_container,
                                               ScaledDotProduct(), Linear(d_model, d_model))
        self.linear1 = Linear(d_model, dim_feedforward)
        self.norm1 = LayerNorm(d_model)

        # self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model) 
        self.norm2 = LayerNorm(d_model)
        # self.dropout1 = Dropout(dropout)
        # self.dropout2 = Dropout(dropout)

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu
        else:
            raise RuntimeError("only relu/gelu are supported, not {}".format(activation))

    def init_weights(self):
        self.mha.in_proj_container.query_proj.init_weights()
        self.mha.in_proj_container.key_proj.init_weights()
        self.mha.in_proj_container.value_proj.init_weights()
        self.mha.out_proj.init_weights()
        self.linear1.weight.data.normal_(mean=0.0, std=0.02)
        self.linear2.weight.data.normal_(mean=0.0, std=0.02)
        self.norm1.bias.data.zero_()
        self.norm1.weight.data.fill_(1.0)
        self.norm2.bias.data.zero_()
        self.norm2.weight.data.fill_(1.0)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, attn_output_weights = self.mha(src, src, src, attn_mask=src_mask)
        src = src + attn_output # self.dropout1(attn_output)
        src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.linear2(self.activation(self.linear1(src)))
        src = src + src2 # self.dropout2(src2)
        src = self.norm2(src)
        return src

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class SelfTransformerEncoder(nn.Module): 
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(SelfTransformerEncoder, self).__init__() 
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm 
    
    def forward(self, src, mask=None, src_key_padding_mask=None): 
        output = src

        for mod in self.layers: 
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
    

class BertModel(nn.Module):
    """Contain a transformer encoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(BertModel, self).__init__()
        self.model_type = 'Transformer'
        self.bert_embed = BertEmbedding(ntoken, ninp)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = SelfTransformerEncoder(encoder_layers, nlayers) # TODO 
        self.ninp = ninp

    def forward(self, src, token_type_input):
        src = self.bert_embed(src, token_type_input)
        output = self.transformer_encoder(src)
        return output


class MLMTask(nn.Module):
    """Contain a transformer encoder plus MLM head."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(MLMTask, self).__init__()
        self.bert_model = BertModel(ntoken, ninp, nhead, nhid, nlayers, dropout=0.5)
        self.mlm_span = Linear(ninp, ninp)
        self.activation = F.gelu
        self.norm_layer = LayerNorm(ninp, eps=1e-12)
        self.mlm_head = Linear(ninp, ntoken)

    def forward(self, inputs): 
        # trans_inputs, token_type_input = split_input(inputs) 
        token_type_input = inputs[1] 
        trans_inputs = torch.transpose(inputs[0], 0, 1)
        
        output = self.bert_model(trans_inputs, token_type_input) 
        output = self.mlm_span(output)
        output = self.activation(output)
        output = self.norm_layer(output)
        output = self.mlm_head(output)
        return output


class NextSentenceTask(nn.Module):
    """Contain a pretrain BERT model and a linear layer."""

    def __init__(self, bert_model):
        super(NextSentenceTask, self).__init__()
        self.bert_model = bert_model
        self.linear_layer = Linear(bert_model.ninp,
                                   bert_model.ninp)
        self.activation = nn.Tanh()
        self.ns_span = Linear(bert_model.ninp, 2)
        

    def forward(self, inputs):
        token_type_input = inputs[1] 
        trans_inputs = torch.transpose(inputs[0], 0, 1)
        # src = src.transpose(0, 1)  # Wrap up by nn.DataParallel
        output = self.bert_model(trans_inputs, token_type_input)
        # Send the first <'cls'> seq to a classifier
        output = self.activation(self.linear_layer(output[0]))
        output = self.ns_span(output)
        return output


class QuestionAnswerTask(nn.Module):
    """Contain a pretrain BERT model and a linear layer."""

    def __init__(self, bert_model):
        super(QuestionAnswerTask, self).__init__()
        self.bert_model = bert_model
        self.activation = F.gelu
        self.qa_span = Linear(bert_model.ninp, 2)

    def forward(self, inputs): 
        src = inputs[0]
        token_type_input = inputs[1]
        output = self.bert_model(src, token_type_input)
        # transpose output (S, N, E) to (N, S, E)
        output = output.transpose(0, 1)
        output = self.activation(output)
        pos_output = self.qa_span(output)
        start_pos, end_pos = pos_output.split(1, dim=-1)
        start_pos = start_pos.squeeze(-1)
        end_pos = end_pos.squeeze(-1)
        return start_pos, end_pos

