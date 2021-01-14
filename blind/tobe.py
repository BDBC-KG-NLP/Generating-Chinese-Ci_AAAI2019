def encode(self, input, padded_sentence, length_sequence,decode_length=None):
        
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
        for x in reversed(outs):
            hidden = self.first_sentence_encoder.forward(x, hidden)
        outs[0] = hidden

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
            self.q_dec_outs.append(torch.stack(self.dec_outs))
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
            x, z, mu, logvar = self.variational_inference.draw_z(self.batch_size)
            self.p_z.append(z)
            self.p_mu.append(mu)
            self.p_logvar.append(logvar)
            self.p_dec_outs.append(x)
        if decode_length == None:
            decode_length = len(self.dec_outs)
        for i in range(1, decode_length):

            x = torch.cat([x,z],1)
            x, z, mu, logvar = self.variational_inference.forward(x)
            self.p_z.append(z)
            self.p_mu.append(mu)
            self.p_logvar.append(logvar)
            self.p_dec_outs.append(x)

        self.p_mu = torch.stack(self.p_mu)
        self.p_logvar = torch.stack(self.p_logvar)
        for dec_outs in self.q_dec_outs:
            outs=[]
            hidden_can = torch.cat(dec_outs)
            content = torch.index_select(hidden_can, 0, self.variable(select_index,False,False))
            hidden = self.variational_inference.dec2word_fun(content)
            
            if self.vowel_tone_input == 'NONE':
                for x_hat in decode_input:

                    output,hidden = self.char_decoder.forward(x_hat, hidden,content)
                    outs.append(output)
            else:
                for i,(x_hat, t, v) in enumerate(zip(decode_input, tone_index, vowel_index)):

                    t = self.variable(one_hot(t, self.tone_dim))
                    if isinstance(v,list):
                        temp = []
                        for vv in v:
                            v = self.variable(one_hot(v, self.vowel_dim))
                            atten_w2v = None
                            if self.is_attention:
                                atten_w2v = self.attention_w2v(decode_input[:i+1])
                            
                            output,hidden = self.char_decoder.forward(x_hat, hidden, content,t, v,atten_w2v)
                            output = self.char_decoder.output_projection(output)
                            temp.append(output)
                        # number of vv , vocab
                        output = torch.mean(torch.cat(temp),0)

                    else:
                        v = self.variable(one_hot(v, self.vowel_dim))
                        # print('13.5 vowel_size:[flatten_num_of_sentences,17]', type(v),v.size())
                        atten_w2v = None
                        if self.is_attention:
                            atten_w2v = self.attention_w2v(decode_input[:i+1])
                        
                        output,hidden = self.char_decoder.forward(x_hat, hidden, content,t, v,atten_w2v)
                        outs.append(self.char_decoder.output_projection(output))
            
            return outs, [self.char_decoder.argmax_logits(out) for out in outs]
            