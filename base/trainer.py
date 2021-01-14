import time

import torch
import torch.nn as nn
from torch.nn.init import orthogonal,uniform,xavier_normal,xavier_uniform
import models as m
from torch.autograd import Variable
import glob
import os
import copy
import statistics_hardcoded
from evaluation import *
from utils import create_model_abbr, lookup
from tensorboardX import SummaryWriter
from rules import random_yunshe
from functools import reduce
import time


import ioHelper
import random

def tran(a):
    row = len(a)
    col = len(a[0])
    # print(col,row)
    b = []
    for x in range(col):
        c = []
        for y in range(row):
            c.append(a[y][x])
        b.append(c)
    return b


vocab_lookup, vowel_lookup, tone_lookup = lookup()
vowel = ioHelper.load("./Data/vowel.json")
cuml = rules.yunshe_statistics(vowel)


yun_stat_onehot= [0.03555043186137205, 0.1400235970134467, 0.063320694220533, 0.032121839846529346, 
                  0.08284893408147274, 0.027628680278480786, 0.07843070854085796, 0.05577271212544244, 
                  0.024786591132206322, 0.09202871270185804, 0.028394952760940638, 0.009467265898712973, 
                  0.08457818086840613, 0.07266150894795147, 0.11503544055263845, 0.057349749169150956, 0.0]

def weights_init(m):
    r""" initialize weights for the whole model w.r.t layer or unit.
    currently default init for each unit except **GRU unit**, we use orthogonal_init

    Args:
        m: an instance of a typical unit / layer
    """

    # if isinstance(m, nn.Linear):
    #     # orthogonal(m.weight.data)
    # # elif isinstance(m,nn.Embedding):
    #     # uniform(m.weight.data,-0.001,0.001)
    #     xavier_normal(m.weight.data)
    # if isinstance(m, nn.GRUCell):
    #     orthogonal(m.weight_ih.data)
    #     orthogonal(m.weight_hh.data)
    pass




def yunjiao_location_helper(yunjiao_list,length_sequence):
    # d[第几个字]=(第几句话，第几个韵摄)
    cum = 0
    d = {}
    for i,yuns in enumerate(yunjiao_list,):
        if i>0:
            cum += length_sequence['ci'][i-1]
        for key,values in yuns.items():
            for v in values:
                d_key = length_sequence['sentence'][cum + v]-2
                if d_key in d:
                    d[d_key].append((cum + v,key))
                else:
                    d[d_key] = [(cum + v,key)]
    return d


class Vrae(nn.Module):
    r"""
    Vrae model, the higher and abstract API for The SongCi project
    """
    def __init__(self,
                 config,
                 step):

        super(Vrae, self).__init__()

        self.step = step
        self.model_sub_path = create_model_abbr(config)
        self.vocab_dim = config.vocab_dim  # default:5956
        self.char_emb_dim = config.char_emb_dim  #
        self.char_hidden_dim = config.char_hidden_dim 
        self.sentence_hidden_dim = config.sentence_hidden_dim
        self.latent_dim = config.latent_dim
        self.kl_min = config.kl_min
        self.is_train = config.is_train

        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.weight_decay = config.weight_decay
        self.clip_norm = config.clip_norm
        self.optimizer = config.optimizer

        self.batch_size = config.batch_size
        self.validate_size = config.validate_size

        self.sample_times = config.sample_times
        self.using_first_sentence = config.using_first_sentence

        self.load_path = config.load_path
        # self.model_dir = config.model_dir
        self.start_step = 0
        self.start_epoch = 0
        self.log_step = config.log_step
        self.max_step = config.max_step
        self.max_epochs = config.max_epochs
        self.save_step = config.save_step
        # self.is_vowel_tone = config.is_vowel_tone
        self.vowel_tone_input = config.vowel_tone_input
        self.vowel_dim = config.vowel_dim
        self.tone_dim = config.tone_dim
        self.is_attention = config.is_attention
        self.is_dec_embedding = config.is_dec_embedding

        self.is_guided =  config.is_guided
        self.is_fixed_p = config.is_fixed_p

        self.build_model()

        self.train_same_as_test = config.train_same_as_test

        # self.reconstruct_loss = nn.NLLLoss(ignore_index=0)
        self.reconstruct_loss = nn.CrossEntropyLoss(ignore_index=0)
        
        self.writer = SummaryWriter(log_dir=self.model_sub_path)

        torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_norm)

        self.is_gpu = config.is_gpu
        if self.is_gpu:
            # self.embedding.cuda()
            # self.char_encoder.cuda()
            # self.sentence_encoder.cuda()
            # self.first_sentence_encoder.cuda()
            # self.variational_inference.cuda()
            # self.char_decoder.cuda()
            self.cuda()
        # torch.backend.cudnn.enabled = False
        self.output_path = config.output_path


        # self.paras = chain(self.embedding.parameters(),
        #                    self.char_encoder.parameters(),
        #                    self.sentence_encoder.parameters(),
        #                    self.first_sentence_encoder.parameters(),
        #                    self.char_decoder.parameters()
        #                    )

    def one_hot(self,index, dim):
        r""" fills a one-hot tensor given index 2d list and dimension

        Args:
            m: an instance of a typical unit / layer
        """
        yun_stat_onehot_index = 17
        tensor = torch.zeros(len(index), dim)
        for li,l in enumerate(index):
            if isinstance(l,list):
                for ll in l:
                    tensor[li][ll] += 1./len(l)
            elif l == yun_stat_onehot_index:
                if self.is_gpu:
                    tensor[li] = torch.cuda.FloatTensor(yun_stat_onehot)
                else:
                    tensor[li] = torch.FloatTensor(yun_stat_onehot)
                
            else:
                tensor[li][l] = 1
        return tensor


    def one_hot_y(self,index, dim):
        r""" fills a one-hot tensor given index 2d list and dimension

        Args:
            m: an instance of a typical unit / layer
        """
        yun_stat_onehot_index = 16
        tensor = torch.zeros(len(index), dim)
        for li,l in enumerate(index):
            # if isinstance(l,list):
            #     for ll in l:
            #         tensor[li][ll] += 1./len(l)
            if l == yun_stat_onehot_index:
                tensor[li][l] = 1
            else:
                tensor[li] = torch.cuda.FloatTensor(yun_stat_onehot)
        return tensor
    def one_hot_yy(self,index, dim,yunjiao=False):
        r""" fills a one-hot tensor given index 2d list and dimension

        Args:
            m: an instance of a typical unit / layer
        """
        tensor = torch.zeros(len(index), dim)
        if yunjiao or index[0] == 16:
            tensor[0][index[0]] = 1
        else:
            # print('index[0]',index[0])
            tensor[0] = torch.cuda.FloatTensor(yun_stat_onehot)
        # tensor[0][index[0]] = 1
        return tensor
    # def one_hot1_(index, dim):
    #     r""" fills a one-hot tensor given index 2d list and dimension

    #     Args:
    #         m: an instance of a typical unit / layer
    #     """
    #     yun_stat_onehot_index = 17
    #     tensor = torch.zeros(1, dim)

    #     print('index',index)
    #     if index == 17:
    #         return torch.cuda.FloatTensor(yun_stat_onehot)
    #     else:
    #         tensor[0][index] = 1.
    #         return tensor
    def build_model(self):
        r"""initialize dimensions(hyper-parameters) and weights for every layer/ unit on model.py
        """

        self.embedding = nn.Embedding(self.vocab_dim, self.char_emb_dim)
        if self.is_dec_embedding:
            self.dec_embedding = nn.Embedding(self.vocab_dim, self.char_emb_dim)

        self.char_encoder = m.CharRNNEncoder(self.embedding,
                                             self.char_emb_dim,
                                             self.char_hidden_dim)

        self.sentence_encoder = m.SentenceRNNEncoder(self.char_hidden_dim,
                                                     self.sentence_hidden_dim)
        self.first_sentence_encoder = m.FirstsentenceRNNEncoder(
            self.sentence_hidden_dim)
        self.variational_inference = m.VariationalInference(self.char_hidden_dim,
            self.sentence_hidden_dim, self.latent_dim)

        if self.is_dec_embedding:
            self.char_decoder = m.CharRNNDecoder(self.dec_embedding,
                                                 self.char_emb_dim,
                                                 self.char_hidden_dim,
                                                 self.sentence_hidden_dim,
                                                 self.vowel_dim,
                                                 self.tone_dim,
                                                 self.vowel_tone_input,
                                                 self.vocab_dim)
        else:
            self.char_decoder = m.CharRNNDecoder(self.embedding,
                                                 self.char_emb_dim,
                                                 self.char_hidden_dim,
                                                 self.sentence_hidden_dim,
                                                 self.vowel_dim,
                                                 self.tone_dim,
                                                 self.vowel_tone_input,
                                                 self.vocab_dim)

        # self.attention = m.Attention(self.char_hidden_dim,self.sentence_hidden_dim)
        # self.attention_simple = m.Attention_simple(self.char_hidden_dim,self.sentence_hidden_dim)
        if self.is_attention:
            self.attention_w2v =m.Attention_w2v(self.char_emb_dim,self.embedding)
        self.apply(weights_init)

    def init_hidden(self,flatten_num_of_sentences):
        r"""initialize each layer's first hidden state, currently using zero initialization
        """
        init = torch.zeros

        self.char_encoder_hidden = self.variable(init(flatten_num_of_sentences, self.char_hidden_dim))

        self.sentence_encoder_hidden = self.variable(init(self.batch_size, self.sentence_hidden_dim))
        self.first_sentence_encoder_hidden = self.variable(init(self.batch_size, self.sentence_hidden_dim))
        self.variational_inference_hidden = self.variable(init(self.batch_size, self.latent_dim))



    def load_model(self):
        r"""load model from **sub_model_path** and find the latest one by regex of the file name Vrae_{epochs}_{steps}.pth
        """
        print("[*] Load models from {}...".format(self.model_sub_path))

        paths = glob.glob(os.path.join(self.model_sub_path, 'Vrae_*_*.pth'))
        paths.sort()

        if len(paths) == 0:
            print("[!] No checkpoint found in {}...".format(self.model_sub_path))
            return

        stepidxes = [int(os.path.basename(path.split('.')[-2].split('_')[-1])) for path in paths]

        step_index = stepidxes.index(max(stepidxes))

        self.start_step = stepidxes[step_index]

        self.start_epoch = int(os.path.basename(paths[step_index].split('.')[-2].split('_')[-2]))

        Vrae_filename = '{}/Vrae_{}_{}.pth'.format(self.model_sub_path, self.start_epoch, self.start_step)
        device = torch.device('cuda:0')
        self.load_state_dict(
            torch.load('{}/Vrae_{}_{}.pth'.format(self.model_sub_path, self.start_epoch, self.start_step),
                       map_location=device))
        """self.load_state_dict(
            torch.load('{}/Vrae_{}_{}.pth'.format(self.model_sub_path, self.start_epoch, self.start_step)))
        """
        self.step._step = self.start_step
        self.step._epoch = self.start_epoch

        print("[*] Model loaded: {}".format(Vrae_filename))

    def encode(self, input, padded_sentence, length_sequence,decode_length=None):
        r""" **encoder workflow**

        Notations:
            - S: flatten 1D list containing sentence length sequence over the batch. (在一批宋词中每首词的句长)
            - C: 1D list containing ci length sequence over the batch. (在一批宋词中每首词的诗长，句数)
            - max(S): max sentence length sequence over the batch. (在一'批'宋词中最大句长数)
            — max(C): max ci length sequence over the batch. (在一'批'宋词中最大词长数(在一批最多句数的词))
            - sum(C): total number of sentence over a batch. (一批中的总句数)
            - B: batch size. (一批宋词的数量(多少首))
        Args:
            - **input** (max(S),sum(C)): a 2d tensor of vocab index after padding, max char sequence as the time major.
            - **padded_sentence** dict['forward','backward'] -> (max(C),B): a 2d tensor of sentence index after padding both for forward and backward char sequence.
            - **length_sequence** dict['ci','sentence']: contains 1d list S and 1d list C

        Flow:
            input ->
            char_level_encdoer ->
            each last char hidden state as each sentence representation ->
            sentence_level_encoder ->
            each sentence_sequence representation(H_enc) for encoder ->
            first_sentence_level_encoder ->
            ci_level representation ->
            replace 'each first sentence_sequence representation' with 'ci_level representation' ->
            the 0th q_z_0 or p_z_0 is zero_inited ->
            loop over i times from 0 to max(C)->
                concat q_z_i with H_enc and concat p_z with H_enc correspondingly ->
                encoder_to_latent_layer ->
                mu,log_var for q_z_i, p_z_i ->
                sample q_z_i(inferenced latent sentence sequence representation) over sample_times; sample p_z_i(assumed truth latent rep z) over 1 time ->
            latent_to_decoder_layter ->
            each sentence_sequence representation(H_dec) for decoder

        Notes:
            if 'using_first_sentence' is true, p_z_1 is using the q_z_1 and first sentence hidden out is exactly same from q_z_1 and p_z_1


        """
        # TODO doc of kl term in math
        # input= [B1,...Bt]
        flatten_num_of_sentences = input.size()[1]
        # print('1.char_enc_input:[最大句长，batchsize * number of sentence]', input.size(),len(input[0]),input.size()[1])
        # print('1.char_enc_input:', torch.sum(input[0]))
        self.init_hidden(flatten_num_of_sentences)
        hidden = self.char_encoder_hidden
        # print('1.5.char_enc_hidden:', torch.sum(hidden[0]))
        outs = []
        min_seq = min(length_sequence['sentence'])
        # print('wwww',len(input))
        # print('inputs',len(input),input[0].size())
        for t, each_t in enumerate(input):
            hidden = self.char_encoder.forward(each_t, hidden)
            # out:[#layer,B*S,hidden_dim]

            if t+1 >= min_seq:

                outs.append(hidden)
        # print('2.char_enc_out:[最大句长-min_seq，batchsize * number of sentence, char_hidden_dim]', input.size()[0]-min_seq+1,len(outs),outs[0].size())
        # print('2.char_enc_out:', torch.sum(outs[0][0]))
        # [B1,...,B sentence]
        # only last hidden vector as one sentence's summary
        # print('sequence_array',sequence_array,len(input),len(sequence_array),len(outs),outs[0].size())
        # print('outs_before',len(outs),outs[0].size(),len(length_sequence['sentence']))
        out = self.get_last_hidden_state(outs, length_sequence['sentence'], min_seq)
        # print('3.char_enc_out_last_hidden:[batchsize * number of sentence, char_hidden_dim]', type(out),len(out),out[0].size())
        # print('3.char_enc_out_last_hidden:', torch.sum(out[0][0]))
        # print('last_hidden_state',len(out),out[0].size())
        # print('outs_after',len(out),out[0].size(),len(length_sequence['sentence']))
        hidden = self.sentence_encoder_hidden
        outs = []
        out, out_reverse = self.padSentence(padded_sentence, out)
        # print('4.char_enc_out_padding:[MAX number of sentence, batchsize,char_hidden_dim]', type(out),len(out),out[0].size())
        # print('padSentence',len(out),out[0].size())
        # print('outs_afterPad',len(out),out[0].size(),len(length_sequence['sentence']))
        for x in out:
            hidden = self.sentence_encoder.forward(x, hidden)
            outs.append(hidden)
        # print('5.sen_enc:[MAX number of sentence, batchsize,sen_hidden_dim]', type(outs),len(outs),outs[0].size())
        hidden = self.first_sentence_encoder_hidden
        for i,x in enumerate(reversed(outs)):
            hidden = self.first_sentence_encoder.forward(x, hidden)
            outs[i] = hidden
        # print('6.after first_sen_enc:[MAX number of sentence, batchsize,sen_hidden_dim]', type(outs),len(outs),outs[0].size())

        # print('outs',len(outs),outs[0].size())
        # hidden = self.first_sentence_encoder_hidden
        # min_seq = min(length_sequence['ci'])
        # outs_reverse = []
        # for t, x in enumerate(out_reverse):
        #     out, hidden = self.first_sentence_encoder.forward(x, hidden)
        #     if t+1 >= min_seq:
        #         outs_reverse.append(out)
        # out = self.get_last_hidden_state(outs_reverse, length_sequence['ci'], min_seq)
        # out = torch.stack(out, 1)
        # outs[0] = out
        
        # print('@@@@@@@@@@@@@@@@@',len(outs),outs[0].size(),outs[1].size())

        self.q_zls=[]
        self.q_muls=[]
        self.q_logvarls=[]

        for i in range(self.sample_times):
            z = self.variational_inference_hidden
            q_z=[]
            q_mu=[]
            q_logvar=[]

            # TODO: temporary solution,need to change later
            self.dec_outs=[]
            self.dec_sen_outs=[]


            for x in outs:
                x = torch.cat([x,z],1)
                # print('6.5,xz:[batchsize,sen_hidden_dim+z_dim]',type(x),x.size())
                out, z, mu, logvar = self.variational_inference.forward(x)
                q_z.append(z)
                q_mu.append(mu)
                q_logvar.append(logvar)
                # out: B, hidden_dim
                self.dec_outs.append(out)
            self.q_zls.append(q_z)
            self.q_muls.append(torch.stack(q_mu))
            self.q_logvarls.append(torch.stack(q_logvar))

        # print('7.lat2dec/dec_outs:[MAX number of sentence, batchsize,sen_hidden_dim]', type(self.dec_outs),len(self.dec_outs),self.dec_outs[0].size())

        self.p_z = []
        self.p_mu = []
        self.p_logvar = []
        self.p_dec_outs = []
        if self.using_first_sentence:
            self.p_z.append(q_z[0])
            self.p_mu.append(q_mu[0])
            self.p_logvar.append(q_logvar[0])
            x = self.dec_outs[0]
            z = self.p_z[0]
            self.p_dec_outs.append(x)
        else:
            x, z, mu, logvar = self.variational_inference.draw_z(self.batch_size,self.is_gpu)
            self.p_z.append(z)
            self.p_mu.append(mu)
            self.p_logvar.append(logvar)
            self.p_dec_outs.append(x)
        if decode_length == None:
            decode_length = len(self.dec_outs)
        for i in range(1, decode_length):
            if not self.is_guided:
                x, z, mu, logvar = self.variational_inference.draw_z(self.batch_size,self.is_gpu)
            else:
                x = torch.cat([x,z],1)
                x, z, mu, logvar = self.variational_inference.forward(x,'p')

            self.p_z.append(z)
            self.p_mu.append(mu)
            self.p_logvar.append(logvar)
            self.p_dec_outs.append(x)
        # print(len(self.dec_outs),len(self.q_zls),len(self.q_zls[0]),len(self.p_z))
        self.p_mu = torch.stack(self.p_mu)
        self.p_logvar = torch.stack(self.p_logvar)
        # print('8.p_logvar:[MAX number of sentence, batchsize,z_dim]', type(self.p_logvar),self.p_logvar.size())

    def encode_same_as_test(self, input, padded_sentence, length_sequence,decode_length=None):
        r""" **this method/approach is discarded since we are not using enough info for the training**

        """
        FIXED_P_DIST = True
        flatten_num_of_sentences = input.size()[1]
        
        self.init_hidden(flatten_num_of_sentences)
        hidden = self.char_encoder_hidden

        outs = []
        min_seq = min(length_sequence['sentence'])

        for t, each_t in enumerate(input):
            hidden = self.char_encoder.forward(each_t, hidden)
            if t+1 >= min_seq:
                outs.append(hidden)
        
        out = self.get_last_hidden_state(outs, length_sequence['sentence'], min_seq)

        hidden = self.sentence_encoder_hidden
        outs = []
        out, out_reverse = self.padSentence(padded_sentence, out)
        
        for x in out:
            hidden = self.sentence_encoder.forward(x, hidden)
            outs.append(hidden)
       
        hidden = self.first_sentence_encoder_hidden
        for i,x in enumerate(reversed(outs)):
            hidden = self.first_sentence_encoder.forward(x, hidden)
            outs[i] = hidden

        self.q_zls=[]
        self.q_muls=[]
        self.q_logvarls=[]

        for i in range(self.sample_times):
            z = self.variational_inference_hidden
            q_z=[]
            q_mu=[]
            q_logvar=[]

            self.dec_outs=[]
            self.dec_sen_outs=[]


            for i,x in enumerate(outs):
                if i == 0:
                    out = x
                x = torch.cat([out,z],1)
                out, z, mu, logvar = self.variational_inference.forward(x)
                q_z.append(z)
                q_mu.append(mu)
                q_logvar.append(logvar)
                self.dec_outs.append(out)
            self.q_zls.append(q_z)
            self.q_muls.append(torch.stack(q_mu))
            self.q_logvarls.append(torch.stack(q_logvar))

        self.p_z = []
        self.p_mu = []
        self.p_logvar = []
        self.p_dec_outs = []

        x, z, mu, logvar = self.variational_inference.draw_z(self.batch_size,self.is_gpu)
        self.p_z.append(z)
        self.p_mu.append(mu)
        self.p_logvar.append(logvar)
        self.p_dec_outs.append(x)
        if decode_length == None:
            decode_length = len(self.dec_outs)
        for i in range(1, decode_length):
            
            x = torch.cat([x,z],1)
            if self.is_fixed_p:
                x, z, mu, logvar = self.variational_inference.forward(x,'p')
            else:
                x, z, mu, logvar = self.variational_inference.forward(x)
            self.p_z.append(z)
            self.p_mu.append(mu)
            self.p_logvar.append(logvar)
            self.p_dec_outs.append(x)

        self.p_mu = torch.stack(self.p_mu)
        self.p_logvar = torch.stack(self.p_logvar)


    def decode(self, decode_input, select_index, tone_index, vowel_index,yunjiao_list,length_sequence):
        r""" **decoder workflow**

        Args:

            - **decode_input** (max(S),sum(C)): a 2d tensor of vocab index after padding, max char sequence as the time major.
            - **select_index**: a 1d list contains valid sentence index on C.
            - **tone_index** (max(S),sum(C)): a 2d list of tone index
            - **vowel_index** (max(S),sum(C)): a 2d list of vowel index

        Flow:

            - each sentence first hidden for decoder ->
            - char_level_decoder ->
            - fill max(S) for each sentence ->
            - greedy_decode(agrmax)->
            - best match index

        Notes:


        """
        # ground_truth_x: [T,B] , [num of char sequence, num of sentence,]
        outs=[]
        # print('dec_outs',len(self.dec_outs),self.dec_outs[0].size())
        if self.is_guided:
            hidden_can = torch.cat(self.dec_outs)
        else:
            hidden_can = torch.cat(self.p_dec_outs)

        # print('9.lat2dec/dec_outs_tensor:[MAX number of sentence, batchsize,sen_hidden_dim]', type(hidden_can),hidden_can.size())

        # print('10.select_index:[1d array]', type(select_index),len(select_index))

        content = torch.index_select(hidden_can, 0, self.variable(select_index,False,False))

        # content = self.attention.context_forward(content,length_sequence)
        # content = self.attention_simple.context_forward(content,length_sequence)

        # content = hidden_can.index_select(0, self.variable(select_index,False,False))
        # content = torch.index_select(hidden_can.data, 0, self.tensor(select_index,False))
        # content = self.variable(content)



        # print('11.context:[flatten_num_of_sentences,sentence_hidden_dim]', type(context),context.size())
        hidden = self.variational_inference.dec2word_fun(content)
        # print('12.dec2word:[flatten_num_of_sentences,char_hidden_dim]', type(hidden),hidden.size())
        # print('13.decode_input:[max char sequence,flatten_num_of_sentences]', type(decode_input),len(decode_input),decode_input[0].size())
        # hiddens = [hidden]
        
        if self.vowel_tone_input == 'NONE':
            for x_hat in decode_input:

                output,hidden = self.char_decoder.forward(x_hat, hidden,content)
                outs.append(output)
        else:
            for i,(x_hat, t, v) in enumerate(zip(decode_input, tone_index, vowel_index)):

                t = self.variable(self.one_hot(t, self.tone_dim))
                v = self.variable(self.one_hot(v, self.vowel_dim))
                # print('13.5 vowel_size:[flatten_num_of_sentences,17]', type(v),v.size())
                atten_w2v = None
                if self.is_attention:
                    atten_w2v = self.attention_w2v(decode_input[:i+1])
                
                output,hidden = self.char_decoder.forward(x_hat, hidden, content,t, v,atten_w2v)
                # print('16!!!!', torch.sum(output[0]),i)
                # hiddens.append(hidden)
                # hidden = self.attention.char_forward(hiddens)
                outs.append(output)
                # if i>0:
                #     hidden = self.attention_simple.hidden_forward(outs)
                
        # print('14.dec_outs:[max char sequence,flatten_num_of_sentences,vocab_dim]', type(outs),len(outs),outs[0].size())
       
        # outs = self.set_yunjiao_const(outs,yunjiao_location_helper(yunjiao_list,length_sequence))

        # print('15.dec_outs_trim_vowel:[max char sequence,flatten_num_of_sentences,vocab_dim]', type(outs),len(outs),outs[0].size())
        # print('15.5 check hidden:[flatten_num_of_sentences,char_hidden_dim]', type(hidden),hidden.size())
        
        # outs = self.attention.char_forward(outs)
        outs = [self.char_decoder.output_projection(out) for out in outs]
        
        return outs, [self.char_decoder.argmax_logits(out) for out in outs]

    #[B1,...Bt]
    def reconstruct(self, decode_input, select_index, tone_index, vowel_index,yunjiao_list,length_sequence):
        r""" same as the 'decode' function
        """
        # ground_truth_x: [T,B] , [num of char sequence, num of sentence,]
        outs=[]

        if self.is_guided:
            hidden_can = torch.cat(self.dec_outs)
        else:
            hidden_can = torch.cat(self.p_dec_outs)
       
        # print('dec',torch.sum(hidden_can))
        # hidden_can = torch.cat(self.dec_outs)
        # temp = torch.index_select(hidden.data, 1, self.tensor(select_index,False))
        # context = self.variable(temp)

        content = torch.index_select(hidden_can, 0, self.variable(select_index,False,False))
        # content = self.attention.context_forward(content,length_sequence)
        hidden = self.variational_inference.dec2word_fun(content)
        sequence = [decode_input[0]]
        atten_w2v = None
        if self.vowel_tone_input == 'NONE':
            for i,x_hat in enumerate(decode_input):
                if i>0:
                    # x_hat = self.variable(torch.from_numpy(self.char_decoder.argmax_logits(out).view(-1)),is_float_type=False)
                    hidden = self.char_decoder.forward(x_hat, hidden)
                outs.append(hidden)
        else:
            i = 0
            for x_hat, t, v in zip(decode_input, tone_index, vowel_index):
                len_v = len(v)
                # if v == yun_stat_onehot_index:
                #     v = self.variable(self.tensor(yun_stat_onehot).view(
                #         1, -1, self.vowel_dim))
                # else:
                # print('v_error',v,'\bdebug:',vowel_index)
                v = self.variable(self.one_hot(v, self.vowel_dim))
                t = self.variable(self.one_hot(t, self.tone_dim))

                if i > 0:
                    # if i ==1:
                    #     v = [random_yunshe(cuml, random.choice(range(cuml[-1]))) for _ in range(len_v)]
                    # else:
                    # TODO
                    if self.vowel_tone_input == 'VOWEL_GRU_TONE_PROJECTION':

                        v = [vowel_lookup[vocab_lookup[_]] for _ in self.char_decoder.argmax_logits(out)[0]]
                        temp = [vocab_lookup[_] for _ in self.char_decoder.argmax_logits(out)[0]]
                        # print(v)
                        # print(temp)
                        v = self.variable(self.one_hot(v, self.vowel_dim))

                    x_hat = self.variable(torch.from_numpy(self.char_decoder.argmax_logits(self.char_decoder.output_projection(out))).view(-1),is_float_type=False)
                  
                    wtf = self.char_decoder.argmax_logits(self.char_decoder.output_projection(out),False).view(-1)
                    sequence.append(wtf)
                    # print(sequence)
                    atten_w2v = None
                    if self.is_attention:
                        atten_w2v = self.attention_w2v(torch.stack(sequence))
                    
                    # print('atten_w2v',atten_w2v.shape)
                    
                out,hidden = self.char_decoder.forward(x_hat, hidden, content,t, v,atten_w2v)
                

                outs.append(out)
                i += 1
        # outs = self.attention.char_forward(outs)
            # outs = self.set_yunjiao_const(outs,yunjiao_location_helper(yunjiao_list,length_sequence))
        print('argmax_ll:\n',self.char_decoder.neg_ll_helper(outs,length_sequence['sentence']))

        return [self.char_decoder.argmax_logits(self.char_decoder.output_projection(out)) for out in outs]
        # return [self.char_decoder.draw_from_softmax(out,sen_len) for out,sen_len in zip(outs,length_sequence['sentence'])]

    # def reconstruct_beam(self, decode_input, select_index, tone_index, vowel_index,yunjiao_list,length_sequence):
    #     r""" same as the 'decode' function
    #     """
    #     # ground_truth_x: [T,B] , [num of char sequence, num of sentence,]
    #     def find_beam_paths(sequence_path,permutation_prob,permutation_index,hidden_list,prob_path,next_x_candiates_index):
    #         pt = permutation_prob.reshape(-1,self.batch_size).argsort(0)
    #         indexff = [[np.unravel_index(j,(beam_size,beam_size))for j in x] for x in pt[-beam_size:]]
    #         hidden_list = [[] for _ in range(beam_size)]
    #         new_s = []
    #         for b_i,b in enumerate(indexff):
    #             temp=[]
    #             for i,j in enumerate(b):
    #                 c = (j[0],j[1],i)
    #                 l = (j[0],i)
    #                 temp.append(sequence_path[l[0]][l[1]]+[permutation_index[c]])
    #                 prob_path[b_i,i] = permutation_prob[c]
    #                 next_x_candiates_index[b_i,i] = permutation_index[c]
    #                 hidden_list[b_i][i].append(j[1])
    #             new_s.append(temp)
            
    #         return new_s,prob_path,next_x_candiates_index,hidden_list


    #     outs=[]
    #     hidden_can = torch.cat(self.p_dec_outs)
    #     context = torch.index_select(hidden_can, 0, self.variable(select_index,False,False))
    #     hidden = self.variational_inference.dec2word_fun(context)
    #     sequence_path = None
    #     prob_path = None
    #     hidden_path = [[hidden]]


    #     for t_i,(x_hat, t, v) in enumerate(zip(decode_input, tone_index, vowel_index)):
    #         v = self.variable(one_hot(v, self.vowel_dim))
    #         t = self.variable(one_hot(t, self.tone_dim))
    #         if t_i ==0:
    #             out,hidden = self.char_decoder.forward(x_hat, hidden, context,t, v)
    #             x_candiates_index,prob_path = self.char_decoder.beams(out,3) # beam,batch
    #             sequence_path = np.expand_dims(x_candiates_index,-1).tolist()
    #             hidden_path.append([hidden])

    #         if i > 0:
    #             permutation_prob = []
    #             permutation_index = []
    #             hidden_list = []

    #             for i,batch in enumerate(x_candiates_index):
    #                 x_hat = self.variable(torch.from_numpy(batch,is_float_type=False))
    #                 if i ==1:
    #                     hidden_previous = hidden_path[-1][0]
    #                 else:
    #                     hidden_previous = hidden_path[-1][hidden_index[0]]
    #                 out,hidden_next = self.char_decoder.forward(x_hat, hidden_previous, context,t, v)
    #                 out = self.char_decoder.output_projection(out)
    #                 next_x_candiates_index,next_x_candiates_prob = self.char_decoder.beams(hidden_next,3) # beam,batch
    #                 next_x_candiates_prob *= prob_path[i]
    #                 hidden_list.append(hidden_next)
    #                 permutation_prob.append(next_x_candiates_prob)
    #                 permutation_index.append(next_x_candiates_index)

    #             sequence_path,prob_path,x_candiates_index =
    #             find_beam_paths(sequence_path,permutation_prob,permutation_index,prob_path,next_x_candiates_index)    
                    
                    


                
    #             out,hidden = self.char_decoder.forward(x_hat, hidden, context,t, v)

    #             outs.append(out)

    #             i += 1


    #     outs = self.attention.char_forward(outs)
    #     outs = [self.char_decoder.output_projection(out) for out in outs]   
    #     return [self.char_decoder.argmax_logits(out) for out in outs]
    #     # return [self.char_decoder.draw_from_softmax(out,sen_len) for out,sen_len in zip(outs,length_sequence['sentence'])]

    
    # def reconstruct1(self, decode_input, select_index, tone_index, vowel_index,yunjiao_list,length_sequence):
    #     r""" same as the 'decode' function
    #     """
    #     # ground_truth_x: [T,B] , [num of char sequence, num of sentence,]
    #     outs=[]
    #     # hidden_can = torch.cat(self.p_dec_outs)
    #     hidden_can = torch.cat(self.p_dec_outs)
    #     # temp = torch.index_select(hidden.data, 1, self.tensor(select_index,False))
    #     # context = self.variable(temp)

    #     content = torch.index_select(hidden_can, 0, self.variable(select_index,False,False))
    #     # content = self.attention.context_forward(content,length_sequence)
    #     hidden = self.variational_inference.dec2word_fun(content)
    #     sequence = [decode_input[0]]
    #     atten_w2v = None
    #     if self.vowel_tone_input == 'NONE':
    #         for i,x_hat in enumerate(decode_input):
    #             # if i>0:
    #                 # x_hat = self.variable(torch.from_numpy(self.char_decoder.argmax_logits(out).view(-1)),is_float_type=False)
    #                 # x_hat = self.variable(torch.from_numpy(self.char_decoder.argmax_logits(self.char_decoder.output_projection(out))).view(-1),is_float_type=False)
    #             out,hidden = self.char_decoder.forward(x_hat, hidden)
    #             outs.append(hidden)
    #     else:
    #         i = 0
    #         for x_hat, t, v in zip(decode_input, tone_index, vowel_index):
    #             len_v = len(v)
    #             # if v == yun_stat_onehot_index:
    #             #     v = self.variable(self.tensor(yun_stat_onehot).view(
    #             #         1, -1, self.vowel_dim))
    #             # else:
    #             v = self.variable(self.one_hot_y(v, self.vowel_dim))
    #             t = self.variable(self.one_hot(t, self.tone_dim))

    #             if i > 0:
    #                 # if i ==1:
    #                 #     v = [random_yunshe(cuml, random.choice(range(cuml[-1]))) for _ in range(len_v)]
    #                 # else:
    #                 # TODO
    #                 if self.vowel_tone_input == 'VOWEL_GRU_TONE_PROJECTION':

    #                     v = [vowel_lookup[vocab_lookup[_]] for _ in self.char_decoder.argmax_logits(out)[0]]
    #                     temp = [vocab_lookup[_] for _ in self.char_decoder.argmax_logits(out)[0]]
    #                     print(v)
    #                     print(temp)
    #                     v = self.variable(self.one_hot(v, self.vowel_dim))

    #                 x_hat = self.variable(torch.from_numpy(self.char_decoder.argmax_logits(self.char_decoder.output_projection(out))).view(-1),is_float_type=False)
                  
    #                 wtf = self.char_decoder.argmax_logits(self.char_decoder.output_projection(out),False).view(-1)
    #                 sequence.append(wtf)
    #                 # print(sequence)
    #                 atten_w2v = None
    #                 if self.is_attention:
    #                     atten_w2v = self.attention_w2v(torch.stack(sequence))
                    
    #                 # print('atten_w2v',atten_w2v.shape)
                    
    #             out,hidden = self.char_decoder.forward(x_hat, hidden, content,t, v,atten_w2v)
                

    #             outs.append(out)
    #             i += 1
    #     # outs = self.attention.char_forward(outs)
    #         # outs = self.set_yunjiao_const(outs,yunjiao_location_helper(yunjiao_list,length_sequence))
    #     # print('argmax_ll:\n',self.char_decoder.neg_ll_helper(outs,length_sequence['sentence']))
    #     outs = [self.char_decoder.output_projection(out) for out in outs]
    #     return outs,[self.char_decoder.argmax_logits(out) for out in outs]

    def reconstruct_batch_size_1_beam(self, decode_input, select_index, tone_index, vowel_index,yunjiao_list,length_sequence):
        r""" same as the 'decode' function
        """
        # ground_truth_x: [T,B] , [num of char sequence, num of sentence,]
        beam_size = 50
        def find_beam_paths(sequence_path,permutation_prob,permutation_index,prob_path,next_x_candiates_index):
            
            permutation_prob = np.array(permutation_prob)
            permutation_index = np.array(permutation_index)

            pt = permutation_prob.reshape(-1).argsort()
            indexff = [np.unravel_index(x,(beam_size,beam_size))for x in pt[-beam_size:]]
            # print('indexff',indexff)
            # hidden_list = [[] for _ in range(beam_size)]
            new_s = []
            hidden_index = []
            for b_i,b in enumerate(indexff):
                
                new_s.append(sequence_path[b[0]]+[permutation_index[b]])
                prob_path[b_i] = permutation_prob[b]
                next_x_candiates_index[b_i] = permutation_index[b]
                hidden_index.append(b[1])
               
            # print(new_s,prob_path,next_x_candiates_index,hidden_index)
            return new_s,prob_path,next_x_candiates_index,hidden_index

        outs=[]
        if self.is_guided:
            hidden_can = torch.cat(self.dec_outs)
        else:
            hidden_can = torch.cat(self.p_dec_outs)
        # print('about_index',hidden_can.size(),select_index)
        content = torch.index_select(hidden_can, 0, self.variable(select_index,False,False))
        # content = self.attention.context_forward(content,length_sequence)
        hidden = self.variational_inference.dec2word_fun(content)


        x_first = decode_input[0]
        tone_index = tone_index.T  #[B,T]
        # vowel_index =vowel_index.T #[B,T]
        vowel_index = tran(vowel_index)
        final = []
        nll = []
        for b_i,(ts, vs) in enumerate(zip(tone_index, vowel_index)):
            hidden_list = []
            for t_i,(t,v) in enumerate(zip(ts,vs)):
                # print('v',v)
                # print(one_hot([v], self.vowel_dim))
                v = self.variable(self.one_hot([v], self.vowel_dim))
                # print('imhere')
                t = self.variable(self.one_hot([t], self.tone_dim))
                if t_i==0:
                    x_hat = x_first[b_i]
                    out,hid = self.char_decoder.forward(x_hat, hidden[b_i].view(1,-1), content[b_i].view(1,-1),t, v)
                    out = self.char_decoder.output_projection(out)
                    x_candiates_index,prob_path = self.char_decoder.beams_1(out,beam_size) # beam,batch
                    prob_path = np.log(prob_path)
                    hidden_list.append([hid])
                    sequence_path = np.expand_dims(x_candiates_index,-1).tolist()
                   
                
                elif t_i > length_sequence['sentence'][b_i]-2: 
                    x_hat = final[-1][-1]
                    x_hat = self.variable(torch.from_numpy(np.array([x_hat])),is_float_type=False)
                    out,hidden_after = self.char_decoder.forward(x_hat, hidden_after.view(1,-1), content[b_i].view(1,-1),t, v)
                    out = self.char_decoder.output_projection(out)
                    out = self.char_decoder.argmax_logits(out)[0]
                    # print('out',out)
                    final[-1].append(out)
                    # print(final[-1])
                else:
                    sub_hidden_list=[]
                    permutation_prob=[]
                    permutation_index=[]
                    for i,x_hat in enumerate(x_candiates_index):
                        atten_w2v = None
                        if self.is_attention:
                            atten_w2v = self.attention_w2v(self.variable(torch.from_numpy(np.array(sequence_path[i])),is_float_type=False).view(-1,1))
                        # print('atten_w2v',atten_w2v.shape)
                       
                        x_hat = self.variable(torch.from_numpy(np.array([x_hat])),is_float_type=False)
                        if t_i == 1:
                            out,hid = self.char_decoder.forward(x_hat, hidden_list[-1][-1].view(1,-1), content[b_i].view(1,-1),t, v)
                        else:
                            out,hid = self.char_decoder.forward(x_hat, hidden_list[-1][hidden_index[i]].view(1,-1), content[b_i].view(1,-1),t, v,atten_w2v)
                        out = self.char_decoder.output_projection(out)
                        next_x_candiates_index,next_x_candiates_prob = self.char_decoder.beams_1(out,beam_size) # beam,batch
                        
                        next_x_candiates_prob = np.log(next_x_candiates_prob) + prob_path[i]
                        sub_hidden_list.append(hid)
                        permutation_prob.append(next_x_candiates_prob)
                        permutation_index.append(next_x_candiates_index)
                    sequence_path,prob_path,x_candiates_index,hidden_index =find_beam_paths(sequence_path,permutation_prob,permutation_index,prob_path,next_x_candiates_index)
                    hidden_list.append(sub_hidden_list)
                    if t_i == length_sequence['sentence'][b_i]-2:
                        best_fit = np.argmax(prob_path)
                        nll.append(prob_path[best_fit])
                        final.append(sequence_path[best_fit])
                        hidden_after = hidden_list[-1][hidden_index[best_fit]]
        print('beam_ll:\n',np.array(nll),'长度:',len(tone_index))
        return np.array(final).T.tolist()



    def loss_function(self,ground_truth_x,reconstruct_x,mask_weights,valid_index):
        # http: // pytorch.org / docs / master / _modules / torch / nn / functional.html
        # https: // discuss.pytorch.org / t / how - can - i - compute - seq2seq - loss - using - mask / 861 / 17?u = xinyuwufei
        # reconstruct_loss=F.cross_entropy(ground_truth_x,reconstruct_x)
        # print('prob...',torch.sum(reconstruct_x[0][0]))

        # reconstruct_x = [torch.log(x) for x in reconstruct_x]
        
        # print('9999............!!!!!!!',len(reconstruct_x),reconstruct_x[0].size())

        reconstruct_x = torch.cat(reconstruct_x)
        # print(reconstruct_x.size(),ground_truth_x.size(),ground_truth_x[0].size())
        ground_truth_x = ground_truth_x.view(-1)
        # print('............!!!!!!!',reconstruct_x.size(),ground_truth_x.size())
        rl_per_char = self.reconstruct_loss(reconstruct_x,ground_truth_x)
        # print('rl_per_char',rl_per_char)
        # print('!!!!!!!',rl_per_char.size())
        total_size = torch.sum(mask_weights)
        total_size += 1e-12
        rl_per_poem = rl_per_char * total_size / self.batch_size



        kl_cost = self.variable(torch.zeros(self.p_mu.size()))

        for j in range(self.sample_times):
            # print('error',self.p_mu.size(),self.q_muls[j].size())
            diff_mu = self.p_mu - self.q_muls[j]

            KL_t = self.p_logvar - self.q_logvarls[j] - 1.0 + \
                   torch.exp(self.q_logvarls[j]) / torch.exp(self.p_logvar) + \
                   (diff_mu ** 2) / torch.exp(self.p_logvar)

            kl_cost += 0.5 * KL_t


        # kl_cost = kl_ave_vector = torch.mean(kl_cost / self.sample_times,0,True)
        kl_cost = kl_ave_vector = (kl_cost / self.sample_times).view(-1,self.latent_dim)

        # print('kl_ave_vector before',kl_ave_vector.size())
        # print('valid_index',valid_index)
        kl_cost = kl_ave_vector = torch.index_select(kl_cost, 0, self.variable(valid_index,False,False))
        # print('kl_ave_vector',kl_ave_vector.size())
        # kl_obj = kl_cost = tf.reduce_sum(kl_ave_vector)


        if self.kl_min:

            # FIXME this is not the normal approach for kl norm clampping, this is the clampping for every dims in Z space.

            kl_ave_vector = torch.max(kl_ave_vector, self.variable([self.kl_min],False))
            

        kl_ave_vector_sum = torch.sum(kl_ave_vector)
        kl_cost_sum = torch.sum(kl_cost)

        # kl_obj_per_char = torch.mean(kl_ave_vector)
        # kl_cost_per_char = torch.mean(kl_cost)

        kl_obj_per_char = kl_ave_vector_sum / total_size
        kl_cost_per_char = kl_cost_sum / total_size
        


        kl_obj_per_poem = kl_ave_vector_sum / self.batch_size
        kl_cost_per_poem = kl_cost_sum / self.batch_size
        # print('kl',kl_obj_per_char,kl_cost_per_char)
        return rl_per_char, rl_per_poem, kl_obj_per_char, kl_obj_per_poem, kl_cost_per_char, kl_cost_per_poem

    def get_last_hidden_state(self, input, sequence_array, min_seq):
        r"""
            - get last char hidden for each single sentence rep via S(applied after char_level_encoder)
            - get first(which is the last for backward input) sentence hidden for each single ci rep via C(applied after
            first_sentence_encoder)
        """
        #[1,120], slicing auto decrease 1 dim
        sequence_array = [i - min_seq for i in sequence_array]
        output = []
        # print('seq_arr',sequence_array,len(input),input[0].size())
        for i, s in enumerate(sequence_array):
            output.append(input[s][i, :])
        # print('omg!!!!!',input[s][i, :].size(),sum(input[s][i, :]))
        return output

    def padSentence(self, padded_sentence, out):
        r""" pad extra sentences, from (sum(C),sentence_hidden) to (max(C)*B,sentence_hidden)
        """
        # torch.cat(temp[i]).view(...) : [1,B*,hidden_dim]
        forward_padding = padded_sentence['forward']
        backward_padding = padded_sentence['backward']
        pad = self.variable(torch.zeros(out[0].size()))

        outs = []
        outs_reverse = []
        # print(pad.size())
        out.append(pad)
        # print('forward_padding[0]',len(forward_padding[0]))
        for i in range(len(forward_padding)):
            temp = []
            temp_reverse = []
            for j in range(len(forward_padding[0])):

                temp.append(out[forward_padding[i][j]])
                temp_reverse.append(out[backward_padding[i][j]])
                # print(sentence_padding[i][j].data.type())
            outs.append(torch.stack(temp))
            outs_reverse.append(torch.stack(temp_reverse))
        # print(outs[0].size())
        return outs, outs_reverse

    def set_yunjiao_const(self, outs, location):
        # location: key is the yunjiao pos of a sentence, value: (number of sen, number of yunshe)
        outs = torch.stack(outs)
        # print('dump',outs.size())
        dump1=np.ones(outs.size(),dtype=np.float32)
        dump2 = np.ones((outs.size()[0],outs.size()[1],1),dtype=np.float32)

        for key, values in location.items():
            for v in values:
                zero_one_mat = np.zeros(self.vocab_dim,dtype=np.float32)
                zero_one_mat[statistics_hardcoded.yunjiao_stat[v[1]]] = 1.
                dump1[key,v[0],:] = zero_one_mat
        outs = outs * Variable(torch.from_numpy(dump1).cuda())

       
        for key, values in location.items():
            for v in values:
               dump2[key][v[0]] =torch.sum(outs[key,v[0],:]).data[0]
        outs = outs / Variable(torch.from_numpy(dump2).cuda())
        return [out for out in outs]

        # for key, values in location.items():
        #     for v in values:
        #         zero_one_mat = np.zeros(self.vocab_dim,dtype=np.float32)
        #         zero_one_mat[statistics_hardcoded.yunjiao_stat[v[1]]] = 1.
        #         zero_one_mat = Variable(torch.from_numpy(zero_one_mat).cuda())
        #         # print('dumpway',len(outs),outs[0].size())
        #         # x = outs[key]
        #         # print('before',torch.sum(outs[key][v[0]]))
        #         outs[key][v[0]] = torch.mul(outs[key][v[0]].clone(),zero_one_mat)
        #         norm = torch.sum(outs[key][v[0]])
        #         outs[key][v[0]] = torch.div(outs[key][v[0]].clone(),norm)
        #         # outs[key].data = x.data
        #         # print('after',torch.sum(outs[key][v[0]]))
        # return outs


    def variable(self, inputs, is_tensor=True, is_float_type=True, is_volatile=False):
        r"""helper function to form a tensor either to a gpu versioned variable or cpu versioned variable
        """
        if not is_tensor:
            inputs = self.tensor(inputs,is_float_type)
        if self.is_gpu:
            with torch.no_grad():
            #out = Variable(inputs.cuda(),volatile=is_volatile)
                out = Variable(inputs.cuda())
        else:
            with torch.no_grad():
                out = Variable(inputs)
            #out = Variable(inputs,volatile=is_volatile)
        return out

    def tensor(self, inputs, is_float_type=True):
        r"""helper function to form a input either to a gpu versioned tensor or cpu versioned tensor with specified
        type[long or float]
        """

        if self.is_gpu:
            prefix = torch.cuda
        else:
            prefix = torch
        if is_float_type:
            t = prefix.FloatTensor
        else:
            t = prefix.LongTensor
        return t(inputs)

    def valid(self, is_init=False):
        r"""validation during training

        """
        if is_init:
            padded_char, self.v_padded_sentence, mask, self.v_length_sequence, self.v_yunjiao_list = \
                self.step.next(False)

            self.v_x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False)
            self.v_x_dec_in = self.variable(torch.from_numpy(padded_char['decode_data']),is_float_type=False)
            self.v_x_dec_ground_truth = self.variable(torch.from_numpy(padded_char['result_data']),is_float_type=False)
            self.v_mask = self.variable(torch.from_numpy(mask))
            # self.sample_x = sample_helper(padded_char['result_data'])
            self.v_tone = padded_char['tone']
            self.v_vowel = padded_char['vowel']
            self.v_result_data = padded_char['result_data']


        else:
            batch_size = self.batch_size
            self.batch_size = self.validate_size
            self.encode(self.v_x, self.v_padded_sentence, self.v_length_sequence)
            x_hat, indexes = self.decode(self.v_x_dec_in, self.v_padded_sentence['valid_index'],
                                         self.v_tone, self.v_vowel, self.v_yunjiao_list,self.v_length_sequence)
            rl_per_char, rl_per_poem, kl_obj_per_char, kl_obj_per_poem, kl_cost_per_char, kl_cost_per_poem = \
                self.loss_function(self.v_x_dec_ground_truth, x_hat, self.v_mask,self.v_padded_sentence['valid_index'])
            l, ls, p, y, rec = eval_length_pingze_yunjiao(self.v_yunjiao_list, self.v_result_data,
                                                 indexes, self.v_length_sequence,True)
            self.write_summary(rl_per_char, kl_obj_per_char, kl_cost_per_char, l, ls, p, y, rec , is_train=False)

            print('====> valid: {} recon loss: {:.4f} KL obj: {:.4f} KL cost: {:.4f} ppl: {:.4f}'
                  .format(self.step._step, float(rl_per_char), float(kl_obj_per_char), float(kl_cost_per_char),float(torch.exp(rl_per_char+kl_cost_per_char))))
            print(l, ls, p, y, rec )

            self.batch_size = batch_size
            # self.sample_x_hat = sample_helper(indexes)
            # reconstruct_with_x(self.sample_x_hat, self.sample_x)


    def train(self):
        r"""training loop

        """
        self.load_model() 

        # if self.optimizer == 'adam':
        #     optimizer = torch.optim.Adam
        # else:
        #     raise Exception('Not implemented')	

        self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.lr,
                betas=(self.beta1, self.beta2),
                weight_decay=self.weight_decay)

        # self.optimizer = torch.optim.SGD(self.parameters(), lr = self.lr, momentum=0.9)
        # self.optimizer = torch.optim.RMSprop(self.parameters(), lr = self.lr, alpha=0.9)

        # https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2

        # model_dict = self.state_dict()
        # for k,v in model_dict.items():
        #     print(k,v.size())

        self.valid(True)

        while (self.step._epoch < self.max_epochs):
            padded_char, padded_sentence, mask, length_sequence, yunjiao_list = self.step.next()
            x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False)
            if self.train_same_as_test:
                self.encode_same_as_test(x, padded_sentence, length_sequence)
            else:
                # print('imhere')
                self.encode(x, padded_sentence, length_sequence)
            x_dec_in = self.variable(torch.from_numpy(padded_char['decode_data']),is_float_type=False)
            x_hat, indexes = self.decode(x_dec_in, padded_sentence['valid_index'], padded_char['tone'], padded_char['vowel'], yunjiao_list, length_sequence)
            
            x_dec_ground_truth = self.variable(torch.from_numpy(padded_char['result_data']),is_float_type=False)
            mask = self.variable(torch.from_numpy(mask))
            rl_per_char, rl_per_poem, kl_obj_per_char, kl_obj_per_poem, kl_cost_per_char, kl_cost_per_poem = \
                self.loss_function(x_dec_ground_truth, x_hat, mask,padded_sentence['valid_index'])
            l, ls, p, y, rec = eval_length_pingze_yunjiao(yunjiao_list, padded_char['result_data'], indexes, length_sequence)
            self.write_summary(rl_per_char, kl_obj_per_char, kl_cost_per_char, l, ls, p, y, rec, is_train=True)
            loss = rl_per_poem + kl_obj_per_poem

            # loss = rl_per_char + kl_obj_per_char

            loss.backward()
            self.optimizer.step()
            self.zero_grad()
            # if self.step._step == 0:
            #     self.writer.add_graph(Vrae(),loss)

            if self.step._step % self.log_step == self.log_step-1:
                print('====> step: {} recon loss: {:.4f} KL obj: {:.4f} KL cost: {:.4f} ppl: {:.4f}'.format(
                    self.step._step, float(rl_per_char), float(kl_obj_per_char), float(kl_cost_per_char), float(torch.exp(rl_per_char+kl_cost_per_char))))
                print(l, ls, p, y, rec )

            if self.step._step % self.save_step == self.save_step - 1 or self.step._epoch==self.start_epoch+1:
                print("[*] Save models to {}...".format(self.model_sub_path))
                torch.save(self.state_dict(), '{}/Vrae_{}_{}.pth'.format(self.model_sub_path, self.step._epoch, self.step._step))
                if self.step._epoch == self.start_epoch + 1:
                    self.start_epoch += 1
                self.valid()

    def test(self):
        r"""test/generating loop

        """
        
        self.load_model()
        self.variational_inference.eval()
        is_random = True
        scores = []
        scores_beam = []
        # for ii in range(1):
        a = [
            ['浣溪沙','浣溪沙'],['浣溪沙','浣溪沙'],['浣溪沙','浣溪沙'],
            ['浣溪沙','西江月'],['浣溪沙','西江月'],['浣溪沙','西江月'],
            ['浣溪沙','清平乐'],['浣溪沙','清平乐'],['浣溪沙','清平乐'],
            ['浣溪沙','菩萨蛮'],['浣溪沙','菩萨蛮'],['浣溪沙','菩萨蛮'],
            ['浣溪沙','忆江南'],['浣溪沙','忆江南'],['浣溪沙','忆江南'],
            ]
        b = ['浣溪沙','浣溪沙']
        for x in (a):
            if is_random:
                # padded_char, padded_sentence, _, length_sequence, yunjiao_list,yunjiao_gt_list = self.step.next_test()
                padded_char,padded_sentence, _,length_sequence, yunjiao_list,yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.next_test_semantic_rule(x)
            else:
                # padded_char, padded_sentence, length_sequence, yunjiao_list = self.step.evaluation()
                padded_char,padded_sentence, _,length_sequence, yunjiao_list,yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.eval_next()
            sentences = padded_char['encode_data'].shape[0]

            x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False,is_volatile=True)
            
            if self.train_same_as_test:
                self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
            else:
                self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
            x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
            
            indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
            indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)

            # temp = eval_length_pingze_yunjiao(yunjiao_list,padded_char['result_data'],indexes, length_sequence,True)
            # print(temp)
            # print('*******************************\n')
            # scores.append(list(temp))

            # x_hat, indexes_t = self.decode(x_dec_in, padded_sentence['valid_index'], padded_char['tone'],
            #                              padded_char['vowel_ground_truth'], yunjiao_list, length_sequence)

            temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

            # print('*******************************\n')
            scores.append(list(temp))
            scores_beam.append(list(temp_beam))
            # print('argmax: ',ii, eval_length_pingze_yunjiao(yunjiao_list, padded_char['result_data'], indexes_r, length_sequence,True))
            # print('train :',ii, eval_length_pingze_yunjiao(yunjiao_gt_list, padded_char['result_data'], indexes, length_sequence))
        # print(scores)
        l,ls,p,y,rec=0,0,0,0,0
        for x in scores:
            l+=x[0]
            ls+=x[1]
            p+=x[2]
            y+=x[3]
            rec+=x[4]
        print('\n\nargmax_final',l/len(scores),ls/len(scores),p/len(scores),y/len(scores),rec/len(scores),len(scores))
        l,ls,p,y,rec=0,0,0,0,0
        for x in scores_beam:
            l+=x[0]
            ls+=x[1]
            p+=x[2]
            y+=x[3]
            rec+=x[4]
        print('beam_final',l/len(scores),ls/len(scores),p/len(scores),y/len(scores),rec/len(scores),len(scores))
    
    
    def same_z_different_cipai(self):
        r"""test/generating loop
        * Z相同下产生不同词牌的词
            一共20个文件分为两组：
                * blind 10 组, 语义独立从 N(0,I) 里生成，用成同一个Z产生5种短词牌的词
                * guided 10组 语义从semantic reference里生成，用同一个Z产生5种短词牌的词

        """
        Z = []
        dec = []
        self.load_model()
        self.variational_inference.eval()
        key = [
            ('鹧鸪天',9),
            ('菩萨蛮',8),
            ('蝶恋花',10),
            ('念奴娇',20),
            ('贺新郎',20),
            ('临江仙',10),
            ('满江红',18),
            ('清平乐',8),
            ('水调歌头',19),
            ('减字木兰花',8)
            ]
        for _ in range(10):
            for i in range(len(key)):
                print('+++++++++++++++++++++++++++++++++++++++++++++++++')   
                padded_char,padded_sentence, _,length_sequence, yunjiao_list,yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.next_test_semantic_rule([key[i],('菩萨蛮',8)])
                print_yuanshi(padded_char['encode_data'])
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')
                x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False,is_volatile=True)
                
                if self.train_same_as_test:
                    self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                else:
                    self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                Z.append(self.dec_outs)
                dec.append(self.q_zls[-1])

                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                
                indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

                _,_,_,_,_,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, _ = self.step.next_test_semantic_rule([key[i],('西江月',8)])
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')
                # if self.train_same_as_test:
                #     self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                # else:
                #     self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                
                indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

                _,_, _,_, _,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, _ = self.step.next_test_semantic_rule([key[i],('忆江南',5)])
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')
                # if self.train_same_as_test:
                #     self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                # else:
                #     self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                
                indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

                _,_, _,_, _,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, _ = self.step.next_test_semantic_rule([key[i],('浣溪沙',6)])
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')

                # if self.train_same_as_test:
                #     self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                # else:
                #     self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')
                
                indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

                _,_, _,_, _,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, _ = self.step.next_test_semantic_rule([key[i],('清平乐',8)])
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')
                # if self.train_same_as_test:
                #     self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                # else:
                #     self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

                _,_, _,_, _,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, _ = self.step.next_test_semantic_rule([key[i],('减字木兰花',8)])
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')
                # if self.train_same_as_test:
                #     self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                # else:
                #     self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

                _,_, _,_, _,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, _ = self.step.next_test_semantic_rule([key[i],('竹枝',4)])
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')
                # if self.train_same_as_test:
                #     self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                # else:
                #     self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

                _,_, _,_, _,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, _ = self.step.next_test_semantic_rule([key[i],('虞美人',8)])
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')
                # if self.train_same_as_test:
                #     self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                # else:
                #     self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
               
                indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

                _,_, _,_, _,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, _ = self.step.next_test_semantic_rule([key[i],('长相思',8)])
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')
                # if self.train_same_as_test:
                #     self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                # else:
                #     self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

                _,_, _,_, _,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, _ = self.step.next_test_semantic_rule([key[i],('杨柳枝',4)])
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')
                # if self.train_same_as_test:
                #     self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                # else:
                #     self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

                _,_, _,_, _,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, _ = self.step.next_test_semantic_rule([key[i],('卜算子',8)])
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')
                # if self.train_same_as_test:
                #     self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                # else:
                #     self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

                # _,_, _,_, _,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, _ = self.step.next_test_semantic_rule([key[i],('如梦令',7)])
                # print('yun: ',yunjiao_list_r)
                # print('vowel: ',padded_char_r['vowel'],'\n')
                # # if self.train_same_as_test:
                # #     self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                # # else:
                # #     self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                # x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                # indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                # indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                # temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

        np.save('dec_X',dec)
        np.save('Z',Z)
    def same_semantic_different_z(self):
        r"""test/generating loop

        """
        
        self.load_model()
        self.variational_inference.eval()
        key = [
            ('浣溪沙',6),
            ('鹧鸪天',9),
            ('菩萨蛮',8),
            ('蝶恋花',10),
            ('念奴娇',20),
            ('贺新郎',20),
            ('临江仙',10),
            ('满江红',18),
            ('清平乐',8),
            ('水调歌头',19),
            # ('减字木兰花',8)
            ]
        for _ in range(5):
            for i in range(len(key)):
                print('+++++++++++++++++++++++++++++++++++++++++++++++++')   
                padded_char,padded_sentence, _,length_sequence, yunjiao_list,yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.next_test_semantic_rule([key[i],('忆江南',5)])

                # print_yuanshi(padded_char['encode_data'])
                x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False,is_volatile=True)
                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                for _ in range(10):
                    if self.train_same_as_test:
                        self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                    else:
                        self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                    
                    _,_, _,_, _,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.next_test_semantic_rule([key[i],('忆江南',5)])
                    indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                    indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                    temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

    def different_semantic_different_z(self):
        r"""this is an example use different 鹧鸪天 to genereate different 清平乐
        """

        Z = []
        dec = []
        self.load_model()
        self.variational_inference.eval()

        for _ in range(100):
            
            print('+++++++++++++++++++++++++++++++++++++++++++++++++')   
            # padded_char,padded_sentence, _,length_sequence, yunjiao_list,yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.next_test_semantic_rule([('鹧鸪天',9),('清平乐',8)])
            # # print_yuanshi(padded_char['encode_data'])
            # print('yun: ',yunjiao_list_r)
            # print('vowel: ',padded_char_r['vowel'],'\n')
            # x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False,is_volatile=True)
            # x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
            # for _ in range(3):
            #     if self.train_same_as_test:
            #         self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
            #     else:
            #         self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
            #     Z.append(self.p_z)
            #     dec.append(self.p_dec_outs)
            #     # _,_, _,_, _,_,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.next_test_semantic_rule([key[i],('忆江南',5)])
            #     indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
            #     indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
            #     temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)
            

            # print('------------------------------------------------') 
            for _ in range(3):
                padded_char,padded_sentence, _,length_sequence, yunjiao_list,yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.next_test_semantic_rule([('鹧鸪天',9),('清平乐',8)])
                x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False,is_volatile=True)
                x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
                print('yun: ',yunjiao_list_r)
                print('vowel: ',padded_char_r['vowel'],'\n')
                if self.train_same_as_test:
                    self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
                else:
                    self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
                Z.append(self.p_z)
                dec.append(self.p_dec_outs)
                indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
                temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)
        np.save(self.model_sub_path+'_dec_X',dec)
        np.save(self.model_sub_path+'Z',Z)
    def random_for_nothing(self):
        r"""this is just for testing will be deleted
        """
        Z = []
        dec = []
        self.load_model()
        self.variational_inference.eval()
        # padded_char, padded_sentence, mask, length_sequence, yunjiao_list = 
        # padded_char,padded_sentence, _,length_sequence, yunjiao_list,yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.next_test_semantic_rule([('鹧鸪天',9),('清平乐',8)])
        padded_char,padded_sentence, _,length_sequence, yunjiao_list,yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.eval_next()
        x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False,is_volatile=True)
        x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
        print('yun: ',yunjiao_list_r)
        print('vowel: ',padded_char_r['vowel'],'\n')
        if self.train_same_as_test:
            self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
        else:
            self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])

        indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
        indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
        temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)
        print(temp_beam)
        print(temp)
    def restore_random_for_nothing(self):
        r"""this is just for testing will be deleted
        """
        self.load_model()
        self.variational_inference.eval()
        padded_char,padded_sentence, _,length_sequence, yunjiao_list,yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.next_test_semantic_rule([('鹧鸪天',9),('忆江南',5)])
        vowel = [[[4, 3], [1, 4], [1, 1], [4, 0], [9, 2]], [[2, 9], [0, 14], [2, 4], [10, 15], [1, 15]], [[14, 12], [4, 2], [14, 5], [3, 4], [14, 9]], [16, [9, 10], [1, 14], [4, 6], [12, 14]], [16, 9, [3, 14], [14, 12], 9], [0, 16, [12, 1], [13, 6], 16], [0, 16, [1, 3], 9, 16], [0, 0, 16, 16, 0], [0, 0, 16, 16, 0]] 
        tone = self.step.padTone(self.step._tone['忆江南'][495])
        
        # decode_data =padded_char_r['padded_char_r']
        metaHolder = []
        
        for j in tone:
            diff = 9 - len(j)
            for k in range(diff):
                #pad append num, current -1, change if needed
                j.append(0)
            metaHolder.append(j)
        metaHolder = np.array(metaHolder).T
        # x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False,is_volatile=True)
        # if self.train_same_as_test:
        #     self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence_r['ci'][0])
        # else:
        #     self.encode(x, padded_sentence, length_sequence,length_sequence_r['ci'][0])
        padded_char_r['tone'] = metaHolder
        padded_char_r['vowel'] = vowel
        yunjiao_list_r = [{9: [1, 3, 4]}]
        z = np.load('z_temp.npy')
        a = z[14]
        b = z[41]
        print(a[0].size())
        a = a.tolist()
        b = b.tolist()

        temp = [self.variational_inference.decode(z.view(1,-1)) for z in a ]
        self.p_dec_outs = temp
        x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
        indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
        indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
        temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

        temp = [self.variational_inference.decode(z.view(1,-1)) for z in b ]
        self.p_dec_outs = temp
        x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
        indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
        indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
        temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)




        # acat=torch.cat(a)
        # bcat=torch.cat(b)
        # diff = acat-bcat
        # diff = diff/5
        # # for i in range(4):
        # #     padded_char,padded_sentence, _,length_sequence, yunjiao_list,yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.next_test_semantic_rule([('鹧鸪天',9),('忆江南',5)])
        # #     acat +=diff

        # #     for z in acat:
        # #         print(z.size())
        # #     temp = [self.variational_inference.decode(z.view(1,-1)) for z in acat ]
        # #     self.p_dec_outs = temp
        # #     x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
        # #     indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
        # #     indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
        # #     temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

        # for z_dim in range(24):
        #     # print('z_dim: ',z_dim)
        #     padded_char,padded_sentence, _,length_sequence, yunjiao_list,yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.next_test_semantic_rule([('鹧鸪天',9),('忆江南',5)])
        #     cum = self.variable(torch.zeros(5,24),is_float_type=False,is_volatile=True)
        #     for i in range(4):
        #         print('z_dim: {} offset: {}'.format(z_dim,i))
        #         # padded_char,padded_sentence, _,length_sequence, yunjiao_list,yunjiao_gt_list,padded_char_r, padded_sentence_r, mask_r, length_sequence_r, yunjiao_list_r, yunjiao_gt_list_r = self.step.next_test_semantic_rule([('鹧鸪天',9),('忆江南',5)])
        #         cum[:,z_dim] += diff[:,z_dim]
        #         # print('debug',z_dim,i,cum)
        #         temp = [self.variational_inference.decode(z.view(1,-1)) for z in (acat+cum) ]
        #         self.p_dec_outs = temp
        #         x_dec_in = self.variable(torch.from_numpy(padded_char_r['decode_data']),is_float_type=False,is_volatile=True)
        #         indexes_r = self.reconstruct(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
        #         indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence_r['valid_index'], padded_char_r['tone'], padded_char_r['vowel'], yunjiao_list_r, length_sequence_r)
        #         temp_beam,temp = eval_wrapper_sr(indexes_b,yunjiao_list_r, padded_char_r['result_data'], indexes_r, length_sequence_r,True)

    # def interpolation(self):
    def Semantic_Latent_Relative(self):
        '''
        This function is designed by Zhaoqing to 
        find if the samiliar pair of z have the similar z' 
        '''
        self.load_model()
        self.variational_inference.eval()
        for i in range(4000):
            x,z,_,_ = self.variational_inference.draw_z(self.batch_size,self.is_gpu)
            print(z)
   
    def blind_z_save(self,x_hat,load_path):
        assert len(x_hat) == len(self.dec_outs)
        z_mean = reduce(lambda x,y:x+y,self.dec_outs) / len(self.dec_outs)
        z_mean = z_mean[0].view(1,-1)
        print(z_mean)
        print(len(z_mean[0]))
        with open(load_path,'a') as f:
            for z in list(z_mean[0]):
                f.write(str(float(z)))
                f.write(' ')
            f.write('\n')
            for x in x_hat:
                f.write(''.join([vocab_lookup[_] for _ in x]))
                f.write(' ')
            f.write('\n\n')
 
           
     
    def Metrical_Performances(self):
        r"""test/generating loop

        """
        
        self.load_model()
        self.variational_inference.eval()
        # self.optimizer = torch.optim.Adam(
        # self.parameters(),
        # lr=self.lr,
        # betas=(self.beta1, self.beta2),
        # weight_decay=self.weight_decay)
        
        scores = []
        scores_beam = []

        fp = open(self.model_sub_path+'_blind_r.csv','w' )
        fps =  open('Eval_Data/Blind/mrcg.txt','w')
        for indexxx in range(4000): # 2585 is calculated for batch_size 32

            padded_char, padded_sentence, mask, length_sequence, yunjiao_list  = self.step.next_generation()
           
            sentences = padded_char['encode_data'].shape[0]

            x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False,is_volatile=True)
            
            if self.train_same_as_test:
                self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence['ci'][0])
            else:
                self.encode(x, padded_sentence, length_sequence)
            x_dec_in = self.variable(torch.from_numpy(padded_char['decode_data']),is_float_type=False,is_volatile=True)
            
        #    x_hat = self.reconstruct(x_dec_in, padded_sentence['valid_index'], padded_char['tone'], padded_char['vowel'], yunjiao_list, length_sequence)
            # indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence['valid_index'], padded_char['tone'], padded_char['vowel'], yunjiao_list, length_sequence)
            
            indexes_r = self.reconstruct(x_dec_in, padded_sentence['valid_index'], padded_char['tone'], padded_char['vowel'], yunjiao_list, length_sequence)
            # mask = self.variable(torch.from_numpy(mask))
            # x_dec_ground_truth = self.variable(torch.from_numpy(padded_char['result_data']),is_float_type=False)
            # print('x_dec_ground_truth',x_dec_ground_truth)
            # rl_per_char, rl_per_poem, kl_obj_per_char, kl_obj_per_poem, kl_cost_per_char, kl_cost_per_poem = \
            #     self.loss_function(x_dec_ground_truth, x_hat, mask,padded_sentence['valid_index'])
            # print('666',rl_per_char, rl_per_poem, kl_obj_per_char, kl_obj_per_poem, kl_cost_per_char, kl_cost_per_poem)
            # print('{} recon loss: {:.4f} KL obj: {:.4f} KL cost: {:.4f} ppl: {:.4f}'
            #       .format(indexxx, float(rl_per_char), float(kl_obj_per_char), float(kl_cost_per_char),float(torch.exp(rl_per_char+kl_cost_per_char))))
            flatten_x = padded_char['result_data']
            flatten_x_hat = indexes_r
            argmax = eval_length_pingze_yunjiao(yunjiao_list,flatten_x,flatten_x_hat, length_sequence)
    
            
            xs,x_hats = packing(flatten_x, flatten_x_hat, length_sequence)
                 
            reconstruct_with_x_adv(x_hats[0],xs[0], yunjiao_list[0],self.output_path)
          #  reconstruct_with_x_abs(x_hats[0],xs[0])
            self.blind_z_save(x_hats[0],'Eval_Data/Blind/mrcg_new.txt')
            # print('*******************************\n')
            scores.append(list(argmax))
            # cores_beam.append(list(temp_beam))s
            l=argmax[0]
            ls=argmax[1]
            p=argmax[2]
            y=argmax[3]
            rec=argmax[4]
            print(l, ls, p, y, rec)
            fp = self.write_summary_ppl(list(argmax),fp)
            # print('argmax: ',ii, eval_length_pingze_yunjiao(yunjiao_list, padded_char['result_data'], indexes_r, length_sequence,True))
            # print('train :',ii, eval_length_pingze_yunjiao(yunjiao_gt_list, padded_char['result_data'], indexes, length_sequence))
        # print(scores)
        l,ls,p,y,rec=0,0,0,0,0
        for x in scores:
            l+=x[0]
            ls+=x[1]
            p+=x[2]
            y+=x[3]
            rec+=x[4]
        print('\n\nargmax_final',l/len(scores),ls/len(scores),p/len(scores),y/len(scores),rec/len(scores),len(scores))
        final = [l/len(scores),ls/len(scores),p/len(scores),y/len(scores),rec/len(scores),len(scores)]
        fp = self.write_summary_ppl(final,fp)
        # l,ls,p,y,rec=0,0,0,0,0
        # for x in scores_beam:
        #     l+=x[0]
        #     ls+=x[1]
        #     p+=x[2]
        #     y+=x[3]
        #     rec+=x[4]
        # print('beam_final',l/len(scores),ls/len(scores),p/len(scores),y/len(scores),rec/len(scores),len(scores))
        # self.writer.export_scalars_to_json(self.model_sub_path + "/all_scalars_test.json")
        # self.writer.close()

    def Generate_ci(self,cipai):
        r"""test/generating ci accorcing to cipai

        """


        self.load_model()
        self.variational_inference.eval()




        padded_char, padded_sentence, mask, length_sequence, yunjiao_list = self.step.get_one_info(cipai)


        x = self.variable(torch.from_numpy(padded_char['encode_data']), is_float_type=False, is_volatile=True)



        if self.train_same_as_test:
            self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence['ci'][0])
        else:
            self.encode(x, padded_sentence, length_sequence)

        x_dec_in = self.variable(torch.from_numpy(padded_char['decode_data']), is_float_type=False,
                                 is_volatile=True)


        indexes_r = self.reconstruct(x_dec_in, padded_sentence['valid_index'], padded_char['tone'],
                                     padded_char['vowel'], yunjiao_list, length_sequence)


        flatten_x = padded_char['result_data']
        flatten_x_hat = indexes_r


        xs, x_hats = packing(flatten_x, flatten_x_hat, length_sequence)

        ci = reconstruct_with_x_adv_new(x_hats[0], xs[0], yunjiao_list[0])
        print(ci)
        sentences = ci.split('\n')
        print(sentences)
        length = int((len(sentences)-1)/2)
        print(length)
        new_ci = ''.join(sentences[:length])+'\n'+''.join(sentences[length:])+'\n'
        result = {}
        result['cipai'] = cipai
        result['ci'] = new_ci
        print("词牌："+cipai)
        print('=======词========')
        print(new_ci)
        return result




    def get_train_ppl1(self):
        r"""
        this is an attempt for calculating 'real'
        currently not using
        """
        
        self.load_model()
        self.variational_inference.eval()
        self.optimizer = torch.optim.Adam(
        self.parameters(),
        lr=self.lr,
        betas=(self.beta1, self.beta2),
        weight_decay=self.weight_decay)
        
        for indexxx in range(2585):
            padded_char, padded_sentence, mask, length_sequence, yunjiao_list  = self.step.next_generation()
           
            sentences = padded_char['encode_data'].shape[0]

            x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False,is_volatile=True)
            
            
            x_dec_in = self.variable(torch.from_numpy(padded_char['decode_data']),is_float_type=False,is_volatile=True)
            mask = self.variable(torch.from_numpy(mask))
            x_dec_ground_truth = self.variable(torch.from_numpy(padded_char['result_data']),is_float_type=False)
            
            final, indexes_r = self.encode_for_real_ppl(x, padded_sentence, length_sequence,x_dec_in, padded_sentence['valid_index'], padded_char['tone'], padded_char['vowel'])
            rl_per_char, kl_obj_per_char, kl_cost_per_char = self.about_loss_for_real_ppl(final,x_dec_ground_truth, mask,padded_sentence['valid_index'])
            print(rl_per_char, kl_obj_per_char, kl_cost_per_char)
            flatten_x = padded_char['result_data']
            flatten_x_hat = indexes_r
            argmax = eval_length_pingze_yunjiao(yunjiao_list,flatten_x,flatten_x_hat, length_sequence,True)

        
    def get_train_ppl(self):
        r"""test/generating loop

        """
        
        self.load_model()
        self.variational_inference.eval()
        # self.optimizer = torch.optim.Adam(
        # self.parameters(),
        # lr=self.lr,
        # betas=(self.beta1, self.beta2),
        # weight_decay=self.weight_decay)
        
        scores = []
        scores_beam = []

        fp = open(self.model_sub_path+'_guided_r.csv','w')
        for indexxx in range(2585):

            padded_char, padded_sentence, mask, length_sequence, yunjiao_list  = self.step.next_generation()
           
            sentences = padded_char['encode_data'].shape[0]

            x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False,is_volatile=True)
            
            if self.train_same_as_test:
                self.encode_same_as_test(x, padded_sentence, length_sequence, length_sequence['ci'][0])
            else:
                self.encode(x, padded_sentence, length_sequence)
            x_dec_in = self.variable(torch.from_numpy(padded_char['decode_data']),is_float_type=False,is_volatile=True)
            
            # x_hat,indexes_r = self.reconstruct(x_dec_in, padded_sentence['valid_index'], padded_char['tone'], padded_char['vowel'], yunjiao_list, length_sequence)
            # indexes_b = self.reconstruct_batch_size_1_beam(x_dec_in, padded_sentence['valid_index'], padded_char['tone'], padded_char['vowel'], yunjiao_list, length_sequence)
            indexes_r = self.reconstruct(x_dec_in, padded_sentence['valid_index'], padded_char['tone'], padded_char['vowel'], yunjiao_list, length_sequence)
            # mask = self.variable(torch.from_numpy(mask))
            # x_dec_ground_truth = self.variable(torch.from_numpy(padded_char['result_data']),is_float_type=False)
            # print('x_dec_ground_truth',x_dec_ground_truth)
            # rl_per_char, rl_per_poem, kl_obj_per_char, kl_obj_per_poem, kl_cost_per_char, kl_cost_per_poem = \
            #     self.loss_function(x_dec_ground_truth, x_hat, mask,padded_sentence['valid_index'])
            # print('666',rl_per_char, rl_per_poem, kl_obj_per_char, kl_obj_per_poem, kl_cost_per_char, kl_cost_per_poem)
            # print('{} recon loss: {:.4f} KL obj: {:.4f} KL cost: {:.4f} ppl: {:.4f}'
            #       .format(indexxx, float(rl_per_char), float(kl_obj_per_char), float(kl_cost_per_char),float(torch.exp(rl_per_char+kl_cost_per_char))))
            flatten_x = padded_char['result_data']
            flatten_x_hat = indexes_r
            argmax = eval_length_pingze_yunjiao(yunjiao_list,flatten_x,flatten_x_hat, length_sequence)
    
            
            xs,x_hats = packing(flatten_x, flatten_x_hat, length_sequence)
        
            reconstruct_with_x_adv(x_hats[0],xs[0], yunjiao_list[0])
            

            # print('*******************************\n')
            scores.append(list(argmax))
            # cores_beam.append(list(temp_beam))s
            l=argmax[0]
            ls=argmax[1]
            p=argmax[2]
            y=argmax[3]
            rec=argmax[4]
            print(l, ls, p, y, rec)
            fp = self.write_summary_ppl(list(argmax),fp)
            # print('argmax: ',ii, eval_length_pingze_yunjiao(yunjiao_list, padded_char['result_data'], indexes_r, length_sequence,True))
            # print('train :',ii, eval_length_pingze_yunjiao(yunjiao_gt_list, padded_char['result_data'], indexes, length_sequence))
        # print(scores)
        l,ls,p,y,rec=0,0,0,0,0
        for x in scores:
            l+=x[0]
            ls+=x[1]
            p+=x[2]
            y+=x[3]
            rec+=x[4]
        print('\n\nargmax_final',l/len(scores),ls/len(scores),p/len(scores),y/len(scores),rec/len(scores),len(scores))
        final = [l/len(scores),ls/len(scores),p/len(scores),y/len(scores),rec/len(scores),len(scores)]
        fp = self.write_summary_ppl(final,fp)
        # l,ls,p,y,rec=0,0,0,0,0
        # for x in scores_beam:
        #     l+=x[0]
        #     ls+=x[1]
        #     p+=x[2]
        #     y+=x[3]
        #     rec+=x[4]
        # print('beam_final',l/len(scores),ls/len(scores),p/len(scores),y/len(scores),rec/len(scores),len(scores))
        # self.writer.export_scalars_to_json(self.model_sub_path + "/all_scalars_test.json")
        # self.writer.close()

    def get_train_ppl1(self):
        r"""test/generating loop

        """
        
        self.load_model()
        self.variational_inference.eval()
        self.optimizer = torch.optim.Adam(
        self.parameters(),
        lr=self.lr,
        betas=(self.beta1, self.beta2),
        weight_decay=self.weight_decay)
        
        for indexxx in range(2585):
            padded_char, padded_sentence, mask, length_sequence, yunjiao_list  = self.step.next_generation()
           
            sentences = padded_char['encode_data'].shape[0]

            x = self.variable(torch.from_numpy(padded_char['encode_data']),is_float_type=False,is_volatile=True)
            
            
            x_dec_in = self.variable(torch.from_numpy(padded_char['decode_data']),is_float_type=False,is_volatile=True)
            mask = self.variable(torch.from_numpy(mask))
            x_dec_ground_truth = self.variable(torch.from_numpy(padded_char['result_data']),is_float_type=False)
            
            final, indexes_r = self.encode1(x, padded_sentence, length_sequence,x_dec_in, padded_sentence['valid_index'], padded_char['tone'], padded_char['vowel'])
            rl_per_char, kl_obj_per_char, kl_cost_per_char = self.about_loss(final,x_dec_ground_truth, mask,padded_sentence['valid_index'])
            print(rl_per_char, kl_obj_per_char, kl_cost_per_char)
            flatten_x = padded_char['result_data']
            flatten_x_hat = indexes_r
            argmax = eval_length_pingze_yunjiao(yunjiao_list,flatten_x,flatten_x_hat, length_sequence,True)    

    def encode_for_real_ppl(self, input, padded_sentence, length_sequence,decode_input, select_index, tone_index, vowel_index,decode_length=None):
        
        flatten_num_of_sentences = input.size()[1]
        self.init_hidden(flatten_num_of_sentences)
        hidden = self.char_encoder_hidden
        outs = []
        min_seq = min(length_sequence['sentence'])
        for t, each_t in enumerate(input):
            hidden = self.char_encoder.forward(each_t, hidden)
            if t+1 >= min_seq:
                outs.append(hidden)
        
        out = self.get_last_hidden_state(outs, length_sequence['sentence'], min_seq)
        
        hidden = self.sentence_encoder_hidden
        outs = []
        out, out_reverse = self.padSentence(padded_sentence, out)
        
        for x in out:
            hidden = self.sentence_encoder.forward(x, hidden)
            outs.append(hidden)
        
        hidden = self.first_sentence_encoder_hidden
        for i,x in enumerate(reversed(outs)):
            hidden = self.first_sentence_encoder.forward(x, hidden)
            outs[i] = hidden

        self.q_zls= []
        self.q_muls= []
        self.q_logvarls= []
        self.q_dec_outs= []

        for i in range(self.sample_times):
            z = self.variational_inference_hidden
            q_z=[]
            q_mu=[]
            q_logvar=[]

            self.dec_outs=[]
            # self.dec_sen_outs=[]


            for x in outs:
                x = torch.cat([x,z],1)
                # print('6.5,xz:[batchsize,sen_hidden_dim+z_dim]',type(x),x.size())
                out, z, mu, logvar = self.variational_inference.forward(x)
                q_z.append(z)
                q_mu.append(mu)
                q_logvar.append(logvar)
                # out: B, hidden_dim
                self.dec_outs.append(out)

            self.q_zls.append(q_z)
            self.q_muls.append(torch.stack(q_mu))
            self.q_logvarls.append(torch.stack(q_logvar))
            self.q_dec_outs.append(torch.cat(self.dec_outs))
        # print('7.lat2dec/dec_outs:[MAX number of sentence, batchsize,sen_hidden_dim]', type(self.dec_outs),len(self.dec_outs),self.dec_outs[0].size())
        blind = True
        self.p_z = []
        self.p_mu = []
        self.p_logvar = []
        self.p_dec_outs = []
        if self.using_first_sentence:
            self.p_z.append(q_z[0])
            self.p_mu.append(q_mu[0])
            self.p_logvar.append(q_logvar[0])
            x = self.dec_outs[0]
            z = self.p_z[0]
            self.p_dec_outs.append(x)
        else:
            x, z, mu, logvar = self.variational_inference.draw_z(self.batch_size,self.is_gpu)
            self.p_z.append(z)
            self.p_mu.append(mu)
            self.p_logvar.append(logvar)
            self.p_dec_outs.append(x)
        if decode_length == None:
            decode_length = len(self.dec_outs)
        for i in range(1, decode_length):

            if blind:
                x, z, mu, logvar = self.variational_inference.draw_z(self.batch_size,self.is_gpu)
            else:
                x = torch.cat([x,z],1)
                x, z, mu, logvar = self.variational_inference.forward(x)
            self.p_z.append(z)
            self.p_mu.append(mu)
            self.p_logvar.append(logvar)
            self.p_dec_outs.append(x)

        self.p_mu = torch.stack(self.p_mu)
        self.p_logvar = torch.stack(self.p_logvar)
        
        final = []
        for i,dec_outs in enumerate(self.q_dec_outs): #sample counter
            rl = []
            outs=[]
            hidden_can = dec_outs
            content = torch.index_select(hidden_can, 0, self.variable(select_index,False,False))
            hidden = self.variational_inference.dec2word_fun(content)
            atten_w2v = None
            if self.vowel_tone_input == 'NONE':
                for ii,x_hat in enumerate(decode_input):
                    if ii>0:
                        x_hat = self.variable(torch.from_numpy(self.char_decoder.argmax_logits(out).view(-1)),is_float_type=False)
                        hidden = self.char_decoder.forward(x_hat, hidden)
                    outs.append(hidden)
            else:
                
                for ii,(x_hat, t, v) in enumerate(zip(torch.t(decode_input), tran(tone_index), tran(vowel_index))): # number of sentence T
                    inside_hidden = hidden[ii]
                    inside_content = content[ii]
                    line_outputs = []
                    for xx,tt,vv in zip(x_hat, t, v):
                        if ii == 0:
                            tt = self.variable(one_hot([tt], self.tone_dim))
                            vv = self.variable(one_hot([vv], self.vowel_dim))
                            out,inside_hidden = self.char_decoder.forward(xx, inside_hidden, inside_content,tt, vv,atten_w2v)
                            output = self.char_decoder.output_projection(out)
                        else:
                            xx = self.variable(torch.from_numpy(self.char_decoder.argmax_logits(self.char_decoder.output_projection(out))).view(-1),is_float_type=False)
                            tt = self.variable(one_hot([tt], self.tone_dim))
                            if isinstance(vv,list):
                                temp = []

                                # sample m times where vv should be an array with length of m

                                for vvv in vv:
                                    vvv = self.variable(self.one_hot([vvv], self.vowel_dim))
                                    
                                    out,inside_hidden = self.char_decoder.forward(xx, inside_hidden, inside_content,tt, vvv,atten_w2v)
                                    output = self.char_decoder.output_projection(out)
                                    temp.append(output)
                                
                                output = torch.mean(torch.stack(temp),0)
                                
                            else:
                                vv = self.variable(one_hot([vv], self.vowel_dim))
                                out,inside_hidden = self.char_decoder.forward(xx, inside_hidden, inside_content,tt, vv,atten_w2v)
                                output = self.char_decoder.output_projection(out)
                        
                        line_outputs.append(output)

                    outs.append(torch.stack(line_outputs))
            newouts = torch.transpose(torch.stack(outs),0,1)       
            final.append(newouts)
        return final,[self.char_decoder.argmax_logits(out) for out in newouts]
    def rl_loss_function(self,ground_truth_x,reconstruct_x):
        # abandon
        reconstruct_x = reconstruct_x.contiguous().view(-1,self.vocab_dim)
        ground_truth_x = ground_truth_x.view(-1)
        rl_per_char = self.reconstruct_loss(reconstruct_x,ground_truth_x)

        return rl_per_char
    def kl_loss_function(self,mask_weights,valid_index):
         # abandon
        total_size = torch.sum(mask_weights)
        total_size += 1e-12
        kl_cost = self.variable(torch.zeros(self.p_mu.size()))

        for j in range(self.sample_times):
            # print('error',self.p_mu.size(),self.q_muls[j].size())
            diff_mu = self.p_mu - self.q_muls[j]

            KL_t = self.p_logvar - self.q_logvarls[j] - 1.0 + \
                   torch.exp(self.q_logvarls[j]) / torch.exp(self.p_logvar) + \
                   (diff_mu ** 2) / torch.exp(self.p_logvar)

            kl_cost += 0.5 * KL_t


        # kl_cost = kl_ave_vector = torch.mean(kl_cost / self.sample_times,0,True)
        kl_cost = kl_ave_vector = (kl_cost / self.sample_times).view(-1,self.latent_dim)

        # print('kl_ave_vector before',kl_ave_vector.size())
        # print('valid_index',valid_index)
        kl_cost = kl_ave_vector = torch.index_select(kl_cost, 0, self.variable(valid_index,False,False))
        # print('kl_ave_vector',kl_ave_vector.size())
        # kl_obj = kl_cost = tf.reduce_sum(kl_ave_vector)


        if self.kl_min:
            # print('imhere')
            # print('kl_ave_vector',kl_ave_vector.size())
            kl_ave_vector = torch.max(kl_ave_vector, self.variable([self.kl_min],False))
            # kl_ave_vector = torch.max(kl_ave_vector.data, self.tensor([self.kl_min]))

        kl_ave_vector_sum = torch.sum(kl_ave_vector)
        kl_cost_sum = torch.sum(kl_cost)

        # kl_obj_per_char = torch.mean(kl_ave_vector)
        # kl_cost_per_char = torch.mean(kl_cost)

        kl_obj_per_char = kl_ave_vector_sum / total_size
        kl_cost_per_char = kl_cost_sum / total_size
        

        # print('type!!!!',type(kl_cost_sum ),kl_cost_sum.size(),type(kl_cost_per_char),kl_cost_sum.size())

        kl_obj_per_poem = kl_ave_vector_sum / self.batch_size
        kl_cost_per_poem = kl_cost_sum / self.batch_size
        # print('kl',kl_obj_per_char,kl_cost_per_char)
        return kl_obj_per_char, kl_cost_per_char
    def about_loss_for_real_ppl(self,final,ground_truth_x,mask_weights,valid_index):
         # abandon
        kl_obj_per_char, kl_cost_per_char = self.kl_loss_function(mask_weights,valid_index)
        avg_rl_per_char = []
        for outs in final:
            rl_per_char = self.rl_loss_function(ground_truth_x, outs) 
            avg_rl_per_char.append(rl_per_char)
        return torch.mean(torch.cat(avg_rl_per_char)),kl_obj_per_char,kl_cost_per_char
    def write_summary(self, rl_per_char, kl_obj_per_char, kl_cost_per_char, l, ls, p, y, rec,is_train=True):
        r"""summary for tensorboardX visualization

        """
        if is_train:
            mode = 'train'
        else:
            mode = 'valid'

        self.writer.add_scalar(mode + '/reconstruct_loss', rl_per_char, self.step._step)
        self.writer.add_scalar(mode + '/kl_obj', kl_obj_per_char, self.step._step)
        self.writer.add_scalar(mode + '/kl_cost', kl_cost_per_char, self.step._step)
        self.writer.add_scalar(mode + '/ppl', torch.exp(rl_per_char+kl_cost_per_char), self.step._step)
        self.writer.add_scalars(mode + '/eval', {"length": l,
                                              "length_strict": ls,
                                              "pingze": p,
                                              "yunjiao":y,
                                              "rec_BOW":rec}, self.step._step)
        if self.start_epoch == self.max_epochs:
            self.writer.export_scalars_to_json(self.model_sub_path + "/all_scalars.json")
            self.writer.close()


    def write_summary_ppl(self, scores,fp):
        r"""summary for tensorboardX visualization

        """
        # if is_beam:
        #     mode = 'beam'
        # else:
        #     mode = 'argmax'

        # self.writer.add_scalars(mode + '/eval', {"length": l,
        #                                       "length_strict": ls,
        #                                       "pingze": p,
        #                                       "yunjiao":y,
        #                                       "rec_BOW":rec}, self.step._step)

       
        line = ','.join([str(s) for s in scores])
        fp.write(line)
        fp.write('\n')
        return fp

