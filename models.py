"""Top-level model classes.

Author:
    Chris Chute (chute@stanford.edu)
"""

import layers
import torch
import torch.nn as nn

class BiDAF(nn.Module):
    """Baseline BiDAF model for SQuAD.
    Based on the paper:
    "Bidirectional Attention Flow for Machine Comprehension"
    by Minjoon Seo, Aniruddha Kembhavi, Ali Farhadi, Hannaneh Hajishirzi
    (https://arxiv.org/abs/1611.01603).
    Follows a high-level structure commonly found in SQuAD models:
        - Embedding layer: Embed word indices to get word vectors.
        - Encoder layer: Encode the embedded sequence.
        - Attention layer: Apply an attention mechanism to the encoded sequence.
        - Model encoder layer: Encode the sequence again.
        - Output layer: Simple layer (e.g., fc + softmax) to get final outputs.
    Args:
        word_vectors (torch.Tensor): Pre-trained word vectors.
        hidden_size (int): Number of features in the hidden state at each layer.
        drop_prob (float): Dropout probability.
    """

    def __init__(self, word_vectors, char_vectors, hidden_size,
                 rnn_type='LSTM',
                 self_att = False,
                 rnn_layers=[1,2,2],
                 drop_prob=0.):
        super(BiDAF, self).__init__()
        self.self_att = self_att
        self.char_vect = not char_vectors is None
        if self.char_vect:
            self.emb = layers.EmbeddingCE(word_vectors=word_vectors,
                                    char_vectors=char_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)
            self.enc = layers.RNNEncoder(input_size=2*hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=rnn_layers[0],
                                     rnn_type=rnn_type,
                                     drop_prob=drop_prob)
        else:
            self.emb = layers.Embedding(word_vectors=word_vectors,
                                    hidden_size=hidden_size,
                                    drop_prob=drop_prob)

            self.enc = layers.RNNEncoder(input_size=hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=rnn_layers[0],
                                     rnn_type=rnn_type,
                                     drop_prob=drop_prob)

        self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)
        if self.self_att:
            self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=rnn_layers[1],
                                     rnn_type=rnn_type,
                                     drop_prob=drop_prob)

            self.att2 = layers.SelfAttention(hidden_size=2 * hidden_size,
                                         drop_prob=drop_prob)

            self.mod2 = layers.RNNEncoder(input_size=6 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=rnn_layers[2],
                                     rnn_type=rnn_type,
                                     drop_prob=drop_prob)
            self.out = layers.BiDAFOutput2(hidden_size=hidden_size,
                                      drop_prob=drop_prob,
                                      rnn_type=rnn_type)

        else:
            #self.att = layers.BiDAFAttention(hidden_size=2 * hidden_size,
            #                        drop_prob=drop_prob)
            self.mod = layers.RNNEncoder(input_size=8 * hidden_size,
                                     hidden_size=hidden_size,
                                     num_layers=rnn_layers[1],
                                     rnn_type=rnn_type,
                                     drop_prob=drop_prob)
            self.out = layers.BiDAFOutput(hidden_size=hidden_size,
                                      drop_prob=drop_prob,
                                      rnn_type=rnn_type)

    def forward(self, cw_idxs, cc_idxs, qw_idxs, qc_idxs):
        c_mask = torch.zeros_like(cw_idxs) != cw_idxs
        q_mask = torch.zeros_like(qw_idxs) != qw_idxs
        c_len, q_len = c_mask.sum(-1), q_mask.sum(-1)

        if self.char_vect:
            c_emb = self.emb(cw_idxs,cc_idxs)         # (batch_size, c_len, hidden_size)
            q_emb = self.emb(qw_idxs,qc_idxs)         # (batch_size, q_len, hidden_size)
        else:
            c_emb = self.emb(cw_idxs)         # (batch_size, c_len, hidden_size)
            q_emb = self.emb(qw_idxs)         # (batch_size, q_len, hidden_size)

        c_enc = self.enc(c_emb, c_len)    # (batch_size, c_len, 2 * hidden_size)
        q_enc = self.enc(q_emb, q_len)    # (batch_size, q_len, 2 * hidden_size)

        att = self.att(c_enc, q_enc,
                       c_mask, q_mask)    # (batch_size, c_len, 8 * hidden_size)

        mod = self.mod(att, c_len)        # (batch_size, c_len, 2 * hidden_size)

        if self.self_att:
            att2 = self.att2(att, mod, c_mask) # (batch_size, c_len, 8 * hidden_size)
            mod2 = self.mod2(att2, c_len)      # (batch_size, c_len, 2 * hidden_size)
            out  = self.out(att2, mod2, c_mask)  # 2 tensors, each (batch_size, c_len)
        else:
            out  = self.out(att, mod, c_mask)  # 2 tensors, each (batch_size, c_len)

        return out
