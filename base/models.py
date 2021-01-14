import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np


from torch.autograd import Variable

class Attention_w2v(nn.Module):
    r""" elegent and beautiful
    """
    def __init__(self, emb_dim, embedding):
        super(Attention_w2v, self).__init__()
        self.soft_max = nn.Softmax(1)
        self.theta = nn.Parameter(torch.Tensor(emb_dim, 1)).cuda()
        init.xavier_normal(self.theta)
        self.embedding = embedding

    def forward(self,sequence):
        # sequence: a list of batched word embedding tensor(B,emb_dim) so far where len of list is T
       
        if sequence.shape[0]== 1:
            return self.embedding(sequence[0])
        else:
            # self.embedding(sequence).t()
            sequence = self.embedding(sequence).transpose(0,1) # (B, T, emb_dim)
            # print('sequence',sequence.shape)
            score_vec = sequence @ self.theta # (B,T,1)
            score_vec = self.soft_max(score_vec) #(B,T,1),vertical slicing as softmax input 
            out = torch.bmm(score_vec.transpose(1, 2), sequence) #(B, 1, T) @ (B, T, emb_dim) => (B ,1 ,emb)
            return out.squeeze_(1) #(B ,emb)


class Attention_simple(nn.Module):

    def __init__(self, hidden_state_dim, context_state_dim):
        super(Attention_simple, self).__init__()
        self.soft_max = nn.Softmax(0)
        self.word_theta = nn.Parameter(torch.Tensor(hidden_state_dim, 1).uniform_(-0.001, 0.001)).cuda()
        self.context_theta =  nn.Parameter(torch.Tensor(context_state_dim, 1).uniform_(-0.001, 0.001)).cuda()

    def context_forward(self,context,length_sequence):
        # context: tensor [number of sentences after flattern,context_state_dim]
        # length_sequence['ci']: a list contans each length of ci for a batch
        # return tensor which size is same as context
        start = 0
        outs = []
        for x in length_sequence['ci']:
            outs.append(context[start])
            for j in range(1,x):
                alpha =  self.soft_max(context[start:start+j+1] @ self.context_theta) # [slicing window size,1]
                # print('context_forward',alpha.size())
                outs.append(torch.sum(context[start:start+j+1]*alpha,0,keepdim=False))
            start += x
        o = torch.stack(outs)

        return o

    def hidden_forward(self,hidden_list):
        # hidden_list: a list of tensor[B,H_dim], len of list is T
        # return tensor [B,H_dim]

        # print('hidden_list',hidden_list[0].shape)
        hidden_tensor = torch.stack(hidden_list,1) # [Batch,T,H_dim]
        alphas = self.soft_max(hidden_tensor @ self.word_theta)
        o = torch.sum(hidden_tensor * alphas,1,keepdim=False)
        # print('o',o.shape)
        return o
        

class Attention(nn.Module):

    def __init__(self, hidden_state_dim, context_state_dim):
        super(Attention, self).__init__()
        self.soft_max = nn.Softmax(-1)
        self.soft_max0 = nn.Softmax(0)

        # self.word_level = nn.Linear(hidden_state_dim, hidden_state_dim)
        self.context_level = nn.Linear(context_state_dim, context_state_dim)
        self.tanh = nn.Tanh()
        
        # self.word_theta = nn.Parameter(torch.Tensor(hidden_state_dim, 1).uniform_(-0.1, 0.1)).cuda()
        # self.context_theta =  nn.Parameter(torch.Tensor(context_state_dim, 1).uniform_(-0.1, 0.1)).cuda()

    def context_forward(self,context,length_sequence):
        # context: tensor [number of sentences after flattern,context_state_dim]
        # length_sequence['ci']: a list contans each length of ci for a batch
        # return tensor which size is same as context
        start = 0
        outs = []
        for x in length_sequence['ci']:
            context_slicing = torch.stack(context[start:start+x])
            u = self.tanh(self.context_level(context_slicing))
            for v_t in context[start:start+x]:
                score =  self.soft_max(v_t @ torch.t(u))
                
                outs.append(torch.sum(score.view(-1,1) * context_slicing,0,keepdim=False))
            start += x
        return torch.stack(outs)

    def char_forward(self,char_hiddens):
        # char_hiddens: a list of tenosr[number of sentences after flattern,hidden_state_dim], len(list) is current_char_seq over a batch
        # return list which size is same as char_hiddens
        # print('char_hidden',len(char_hiddens),char_hiddens[0].size())
        max_char_seq = len(char_hiddens)
        char_hiddens = torch.stack(char_hiddens)
        char_hiddens = torch.transpose(char_hiddens,0,1)
        outs = []
        for x in char_hiddens:
            out = []
            for t in x:
                score =  self.soft_max(t @ torch.t(x))
                # print('score_size',score.size(),x.size())
                out.append(torch.sum(score.view(-1,1) * x,0,keepdim=False))
            outs.append(torch.stack(out))

        outs = torch.stack(outs)
        outs = torch.transpose(outs,0,1)
        outs = [x for x in outs]
        return outs

        



class cGRU(nn.Module):
    def __init__(self, emb_dim, hidden_state_dim, context_state_dim):
        super(cGRU, self).__init__()
        self.hidden_state_dim = hidden_state_dim
        self.u_x = nn.Linear(emb_dim, hidden_state_dim)
        self.u_concat_h_context = nn.Linear(hidden_state_dim+context_state_dim, hidden_state_dim)
        # self.u_bias = Variable(torch.zeros(1,hidden_state_dim)).cuda()
        
        self.r_x = nn.Linear(emb_dim, hidden_state_dim)
        self.r_concat_h_context = nn.Linear(hidden_state_dim+context_state_dim, hidden_state_dim)
        # self.r_bias = Variable(torch.zeros(1,hidden_state_dim)).cuda()
        
        self.h_x = nn.Linear(emb_dim, hidden_state_dim)
        self.h_concat_h_context = nn.Linear(hidden_state_dim+context_state_dim, hidden_state_dim)
        


    def forward(self, x,h,c):
        # x = torch.squeeze(x)
        # h = torch.squeeze(h)
        # c = torch.squeeze(c)
        # print('hcsize',h.size(),c.size())
        concat_h_c = torch.cat([h,c],-1)
        # print('okay')
        # print('xsize',x.size(),h.size(),c.size())
        r = F.sigmoid(
            self.r_x(x) 
            + self.r_concat_h_context(concat_h_c)
            )
        # print('rsize',r.size())
        u = F.sigmoid(
            self.u_x(x) 
            + self.u_concat_h_context(concat_h_c)
            )
        h_candidate = F.tanh(
            self.h_x(x) 
            + r * self.h_concat_h_context(concat_h_c)
            )

        new_h = u * h + (1 - u) * h_candidate
        # new_h = new_h.view(1,-1,self.hidden_state_dim)
        # print('new_h',new_h.size())
        return new_h,new_h

class CharRNNEncoder(nn.Module):
    r"""Character-level recurrent encoder cell.

    Args:
        - embedding: character embedding lookup variable, requires_grad.
        - char_emb_dim: the number of expected dimension for character embedding.
        - hidden_dim: the number of expected dimension for encoding a summary of a sequence of chars,
        - n_layers: the number of expected layers for GRU. Default: 1.

    inputs: input, hidden
        - **input**: the length of (batch * sentence_length) 1D array containing char index over the char sequence `c_t`
        - **hidden** (n_layers, batch * sentence_length, hidden_dim): tensor containing the hidden state at char
        sequence `c_t`.

    outputs: output, h'
        - **h'** (n_layers, batch * sentence_length, hidden_dim): tensor containing the hidden state at next char
        sequence `c_t+1`.

    notes:
        - input is transformed to char_embedding by its indexes with (1, batch * sentence_length, char_emb_dim) in
        `forward` function.

    """
    def __init__(self, embedding, char_emb_dim, hidden_dim):
        super(CharRNNEncoder, self).__init__()
        
        self.embedding = embedding
        self.hidden_size = hidden_dim
        self.gru_cell = nn.GRUCell(char_emb_dim, hidden_dim)

    def forward(self, input, hidden):
        num_of_sentences = len(input)
        """embedded = self.embedding(input)"""
        embedded = self.embedding(input.type((torch.cuda.LongTensor)))
        # output = embedded
        # print('embedded',torch.sum(embedded),embedded.size())
        
        hidden = self.gru_cell(embedded, hidden)
        return hidden


class SentenceRNNEncoder(nn.Module):
    r"""sentence-level recurrent encoder cell.

    Args:
        - sentence_emb_dim: the number of expected dimension for sentence encoding.
        - hidden_dim: the number of expected dimension for encoding a summary of a sequence of sentences.

    inputs: input, hidden
        - **input**: the length of (1, batch, sentence_emb_dim) tensor containing a batch of sentences summary over the
        sentence sequence `s_t`.
        - **hidden** (1, batch, hidden_dim): tensor containing the hidden state at sentence sequence `s_t`.

    outputs: output, h'
        - **h'** (1, batch, hidden_dim): tensor containing the hidden state at next sentence sequence `st+1`.

    """
    def __init__(self, char_emb_dim, sentence_emb_dim,):
        super(SentenceRNNEncoder, self).__init__()
        self.gru_cell = nn.GRUCell(char_emb_dim, sentence_emb_dim)

    def forward(self, sentence_inputs, hidden):
        # print('debug:',sentence_inputs.size(),hidden.size())
        hidden = self.gru_cell(sentence_inputs, hidden)
        return hidden


class FirstsentenceRNNEncoder(nn.Module):
    r"""First sentence-level recurrent encoder cell.

    Args:
        - hidden_dim: the number of expected dimension for encoding a summary of a sequence of sentences.

    inputs: input, hidden
        - **sentence_input**: the length of (1, batch, sentence_emb_dim) tensor containing a batch of sentences summary
        over the sentence sequence `s_t'`.
        - **hidden** (1, batch, hidden_dim): tensor containing the hidden state at sentence sequence `s_t'`.

    outputs: output, h'
        - **h'** (1, batch, hidden_dim): tensor containing the hidden state at next time sequence `st'+1`.

    notes:
        - `s_t'`: denotes backward time sequence at `s_t'` which inputs is a reverse sequence list from sentence-level
        recurrent encoder cell.

    """
    def __init__(self, hidden_size):
        super(FirstsentenceRNNEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)

    def forward(self, sentence_input, hidden):
        hidden = self.gru_cell(sentence_input, hidden)
        return hidden


class VariationalInference(nn.Module):
    # a.k.a VAE
    r"""Variational Inference cell.

     Args:
         - sentence_emb_dim: the number of expected dimension for sentence encoding.
         - latent_dim: the number of expected dimension for the latent space of a sequence of sentences.

     inputs: input
         - **input**: the length of (batch, sentence_emb_dim) tensor containing a batch of sentences summary over the
         sentence sequence s_t.

     outputs: decoder_input, z, mu, logvar
         - **decoder_input** (batch, hidden_dim): tensor containing the decoder input(hidden state) at current sentence
         sequence st.
         - **z** (batch, latent_dim): tensor containing latent state at at current sentence sequence st.
         - **mu** (batch, latent_dim): tensor containing means at current sentence sequence st.
         - **logvar** (batch, latent_dim): tensor containing log(variances) at current sentence sequence st.

     """
    def __init__(self, char_hidden_dim, sentence_hidden_dim, latent_dim):
        super(VariationalInference, self).__init__()
        self.latent_dim = latent_dim
        # self.enc2lat = nn.Linear(sentence_hidden_dim + latent_dim, latent_dim*2)
        self.enc2lat = nn.Linear(sentence_hidden_dim + latent_dim, char_hidden_dim)
        self.lat_BN1 = nn.BatchNorm1d(char_hidden_dim)
        self.enc2linear = nn.Linear(char_hidden_dim, latent_dim*2)
        self.lat_BN2 = nn.BatchNorm1d(latent_dim*2)

        self.enc2latq = nn.Linear(sentence_hidden_dim + latent_dim, char_hidden_dim)
        self.lat_BN1q = nn.BatchNorm1d(char_hidden_dim)
        self.enc2linearq = nn.Linear(char_hidden_dim, latent_dim*2)
        self.lat_BN2q = nn.BatchNorm1d(latent_dim*2)
        # self.lat2linear = nn.Linear(latent_dim, latent_dim)

        self.lat2dec = nn.Linear(latent_dim, sentence_hidden_dim)
        self.dec_BN1 = nn.BatchNorm1d(sentence_hidden_dim)
        self.dec2word = nn.Linear(sentence_hidden_dim, char_hidden_dim)
        self.dec_BN2 = nn.BatchNorm1d(char_hidden_dim)
        self.prelu = nn.PReLU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x,mode):
        # mu_logvar = self.prelu(self.enc2lat(x))
        if mode == 'p':
            mu_logvar = self.lat_BN2(self.enc2linear(self.lat_BN1(self.prelu(self.enc2lat(x)))))
        else:
            mu_logvar = self.lat_BN2q(self.enc2linearq(self.lat_BN1q(self.prelu(self.enc2latq(x)))))
        # mu_logvar = self.lat_BN2(self.prelu(self.enc2lat(x)))

        mu, logvar = torch.split(mu_logvar, self.latent_dim, -1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)

    def decode(self, z):
        # l2d = self.prelu(self.lat2dec(self.lat2linear(z)))
        l2d = self.prelu(self.lat2dec(z))
        return self.dec_BN1(l2d)

    def forward(self,x,mode='q'):
        mu, logvar = self.encode(x,mode)
        z = self.reparameterize(mu, logvar)
        l2d = self.decode(z)
        return l2d, z, mu, logvar

    def dec2word_fun(self,l2d):
        # print('l2d',l2d.size())
        d2w = self.prelu(self.dec2word(l2d))
        return self.dec_BN2(d2w)

    def draw_z(self, batch_size,is_gpu):
        '''
        generating initial z from the Gaussian distribution (0,I)
        :param batch_size:
        :return:
        '''
        mu = Variable(torch.zeros(batch_size,self.latent_dim))
        logvar = Variable(torch.ones(batch_size,self.latent_dim))
        if is_gpu:
            mu = mu.cuda()
            logvar = logvar.cuda()

        
        z = torch.normal(mu, logvar)
        decoder_input = self.decode(z)
        return decoder_input, z, mu, logvar


class CharRNNDecoder(nn.Module):
    r"""Character-level recurrent decoder cell.

    Args:
        - embedding: character embedding lookup variable, requires_grad.
        - emb_dim: the number of expected dimension for character embedding (**plus** number of tone and vowel dim if
        - using vowel and tone information).
        - hidden_dim: the number of expected dimension for encoding a summary of a sequence of chars,
        - vocab_dim: the number of vocabulary in corpus.
        - n_layers: the number of expected layers for GRU. Default: 1.

    inputs: input, hidden, tone_embedded(depends), tone_embedded, vowel_embedded(depends)
        - **input**: the length of (batch * sentence_length) 1D array containing char index over the char sequence c_t
        - **hidden** (n_layers, batch * sentence_length, hidden_dim): tensor containing the hidden state at char
        sequence c_t.
        - **tone_embedded** (1,batch * sentence_length,tone_emb_dim): tensor containing the embedding tone info at char
        sequence c_t.
        - **vowel_embedded** (1,batch * sentence_length,tone_emb_dim): tensor containing the embedding vowel info at
        char sequence c_t.

    outputs: output, h'
        -**h'** (n_layers, batch * sentence_length, hidden_dim): tensor containing the hidden state at next char
        sequence c_t+1.

    notes:
        - input is transformed to char_embedding by its indexes with (1, batch * sentence_length, char_emb_dim) in
        forward function.

     """
    # under implement
    # TODO: customized gru if is required

    def __init__(self, embedding, emb_dim, hidden_dim, sentence_hidden_dim, vowel_dim, tone_dim, tone_vowel_input, vocab_dim):

        super(CharRNNDecoder, self).__init__()
        self.embedding = embedding
        self.tone_vowel_input = tone_vowel_input
        aug_emb_dim = emb_dim
        aug_proj_dim = hidden_dim
        if self.tone_vowel_input == 'VOWEL_GRU_TONE_PROJECTION':
            aug_emb_dim += vowel_dim
            aug_proj_dim += tone_dim
        if self.tone_vowel_input == 'DEC_RNN':
            aug_emb_dim += vowel_dim + tone_dim
        if self.tone_vowel_input == 'PROJECTION':
            aug_proj_dim += vowel_dim + tone_dim
        self.gru_cell = nn.GRUCell(aug_emb_dim, hidden_dim)
        # self.gru_cell = cGRU(aug_emb_dim, hidden_dim, sentence_hidden_dim)
        self.soft_max = nn.Softmax(-1)
        self.vocab_dim = vocab_dim
        self.output_projection = nn.Linear(aug_proj_dim, vocab_dim)
        # self.emb_dim = emb_dim
        self.temp =aug_proj_dim
       


    def forward(self, input, hidden,context=None, tone_embedded=None, vowel_embedded=None, use_embedded=None):
        num_of_sentences = len(input)
        if use_embedded is None:
            embedded = self.embedding(input.type((torch.cuda.LongTensor)))
        else:
            embedded = use_embedded
        if self.tone_vowel_input == 'DEC_RNN':
            if tone_embedded is not None:
                embedded = torch.cat([embedded, vowel_embedded, tone_embedded], -1)
                # embedded = torch.cat([embedded, tone_embedded], -1)
        if self.tone_vowel_input == 'VOWEL_GRU_TONE_PROJECTION':
            if tone_embedded is not None:
                # print('wwwww',embedded.size(),vowel_embedded.size())
                embedded = torch.cat([embedded, vowel_embedded], -1)
        output = embedded
        
        # output,hidden = self.gru_cell(output, hidden, context)
        
        hidden = self.gru_cell(output, hidden)
        

        if self.tone_vowel_input == 'PROJECTION':
            output = torch.cat([hidden, vowel_embedded, tone_embedded], -1)
        if self.tone_vowel_input == 'VOWEL_GRU_TONE_PROJECTION':
            output = torch.cat([hidden,tone_embedded], -1)

        # print('out_proj',output.size(),self.temp)
        # output = self.output_projection(output)
        # return self.soft_max(output),hidden
        # return output,hidden
        return hidden,hidden


    def argmax_logits(self, input,is_numpy=True):
        # print('input size',input.size())
        _, index = torch.max(input, -1)
        # print('index size',index.data.cpu().numpy().shape)
        if is_numpy:
            return index.data.cpu().numpy()
        else:
            return index

    def draw_from_softmax(self, input,current_ref_len,is_truncate=True):
        # print('out size',input.data.cpu().view(-1,self.vocab_dim).size())
        prob = self.soft_max(input).data.cpu().view(-1,self.vocab_dim).numpy()
        # print('prob size',prob.shape)
        l = []
        # print('debug',current_ref_len,len(prob))
        for i,x in enumerate(prob):
            if is_truncate:
                if i< current_ref_len-1:
                    l.append(np.random.choice(len(x), p=x))
                else:
                    index = np.argmax(x, -1)
                    
                    l.append(index)
            else:
                l.append(np.random.choice(len(x), p=x))

        return np.array(l)

    def beam_search(self, so_far_sequence, so_far_prob,current_ref_len, beams):
        current_prob = self.soft_max(input).data.cpu().view(self.vocab_dim).numpy()
        current_index = np.argsort(current_prob,-1)[:,-beams:]
        current_prob = [current_prob[i,j] for i,x in enumerate(current_index) for j in x]
        current_prob = [[current_prob[i,j] for j in x] for i,x in enumerate(current_index)]
        if len(so_far_sequence)==0:
            so_far_sequence = [[[i] for i in x] for x in current_index] #B,Beams,Seq
            so_far_prob = current_prob.reshape(-1,beams,1)

        else:

            # permutations = so_far_prob @ np.expand_dims(current_prob,1)
            permutations = np.einsum('abc,acd->abd', so_far_prob, current_prob)
            unravel_shape = permutations.shape
            permutations = permutations.reshape(-1,beams*beams)
            index = np.argsort(permutations,-1)[:,-beams:]
            so_far_prob = permutations[index]
            index = [np.unravel_index(x,unravel_shape) for x in index]
            so_far_sequence = [so_far_sequence[i[0]][i[1]] + [current_index[i[1]]] for i in index]

        return so_far_sequence,so_far_prob

    def beams(self,input,beam):
        # input: B or B,1
        current_prob = self.soft_max(input).data.cpu().view(self.vocab_dim).numpy() # Batch, vocab
        current_index = np.argsort(current_prob,-1)[:,-beam:] # batch,beam
        current_prob = np.array([[current_prob[i,j] for j in x] for i,x in enumerate(current_index)]) # batch,beam
        return current_index.T,current_prob.T

    def beams_1(self,input,beam):
        # input: B or B,1
        current_prob = self.soft_max(input).data.cpu().view(self.vocab_dim).numpy() # Batch, vocab
        current_index = np.argsort(current_prob)[-beam:] # beam
        current_prob = current_prob[current_index]
        # print(current_index,current_prob.shape)
        return current_index,current_prob

    def neg_ll_helper(self,input,sentence_length):
        out = []
        for x in input:
            x =self.output_projection(x)
            p, _ = torch.max(self.soft_max(x), -1)
            
            out.append(p.data.cpu().numpy())
        out = np.array(out).T
        out_p=[0]*len(sentence_length)
        for j,(x,l) in enumerate(zip(out,sentence_length)):
            for i in range(l-1):
                out_p[j] += np.log(x[i])
        return  np.array(out_p)







