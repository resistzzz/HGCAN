
import torch.nn as nn
import torch
import numpy as np
import math
from torch.autograd import Variable
from torch.nn import utils as nn_utils
import torch.nn as nn
import torch
import numpy as np
from aggregator import *
import math
from torch.autograd import Variable
from torch.nn import utils as nn_utils
from torch.nn import Transformer
import torch.nn.functional as F
import random
import torch.nn.functional as F
import random
import time


class SelfAttention1(nn.Module):
    def __init__(self, d_in, d_out):
        super(SelfAttention1, self).__init__()
        ## 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_in, d_out)
        self.W_K = nn.Linear(d_in, d_out)
        self.W_V = nn.Linear(d_in, d_out)
        self.d_in = d_in
        self.layer_norm = nn.LayerNorm(d_out)
        # nn.init.xavier_normal_(self.W_Q.weight)
        # nn.init.xavier_normal_(self.W_K.weight)
        # nn.init.xavier_normal_(self.W_V.weight)

    def forward(self, Q, K, V, attn_mask):
        q_s = self.W_Q(Q)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V)
        scores = torch.matmul(q_s, k_s.transpose(-1, -2)) / np.sqrt(self.d_in)
        scores = scores + attn_mask  # Fills elements of self tensor with value where mask is one.
        attn = torch.softmax(scores, dim=-1)  #[100 ,4, 12, 12]
        output = torch.matmul(attn, v_s)  #[100 ,4 ,12 ,100]

        return output

class Attention(nn.Module):
    def __init__(self, d_in, d_out):
        super(Attention, self).__init__()
        # 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_in, d_out)
        self.W_K = nn.Linear(d_in, d_out)
        self.W_V = nn.Linear(d_in, d_out)
        self.linear = nn.Linear(d_out, 1)
        self.sigmoid = nn.Sigmoid( )
        self.layer_norm = nn.LayerNorm(d_out)
        # nn.init.xavier_normal_(self.W_Q.weight)
        # nn.init.xavier_normal_(self.W_K.weight)
        # nn.init.xavier_normal_(self.W_V.weight)

    def forward(self, Q, K, V, attn_mask):
        q_s = self.W_Q(Q)  # q_s: [bs, 1,d]   [bs, c,1,d]
        k_s = self.W_K(K)  # k_s: [bs,L,d]    [bs, c, L, d]
        # scores = torch.matmul(q_s, k_s.transpose(-1,-2)) / np.sqrt(q_s.size(-1))
        scores = self.linear(self.sigmoid(q_s + k_s))   # [bs, L, 1]  [bs,c,L,1]
        scores = scores * attn_mask
        # scores = scores.masked_fill(attn_mask==0, -1e9)
        # scores = torch.softmax(scores, dim = 1) ## [bs, L, 1]
        output = torch.matmul(scores.transpose(-1, -2), V)
        output = output.squeeze(-2)
        return output    #[bs,d]  [bs, c,d]

        # scores = self.linear(self.sigmoid(q_s + k_s))  # [bs, L, 1]
        # scores = scores + attn_mask.unsqueeze(-1) # [bs, L, 1]
        # attn_scores = torch.softmax(scores, dim=1)
        # output = torch.matmul(attn_scores.transpose(-1, -2), V)
        # output = output.squeeze(-2)
        # return output    #[bs,d]  [bs, c,d]

class Attention1(nn.Module):
    def __init__(self, d_in, d_out):
        super(Attention1, self).__init__()
        # 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_in, d_out)
        self.W_K = nn.Linear(d_in, d_out)
        self.W_V = nn.Linear(d_in, d_out)
        self.linear = nn.Linear(d_out, 1)
        self.sigmoid = nn.Sigmoid( )
        self.layer_norm = nn.LayerNorm(d_out)
        # nn.init.xavier_normal_(self.W_Q.weight)
        # nn.init.xavier_normal_(self.W_K.weight)
        # nn.init.xavier_normal_(self.W_V.weight)

    def forward(self, Q, K, V, attn_mask):
        q_s = self.W_Q(Q)  # q_s:   [bs, c,1,d]
        k_s = self.W_K(K)  # k_s   [bs, c, L, d]
        v_s = self.W_V(V)
        # scores = torch.matmul(q_s, k_s.transpose(-1,-2)) / np.sqrt(q_s.size(-1))
        scores = self.linear(self.sigmoid(q_s + k_s))   #  [bs,c,L,1]
        # attn_mask = torch.where(attn_mask==0, torch.ones_like(attn_mask).to('cuda:0'),torch.zeros_like(attn_mask).to('cuda:0'))
        # scores = scores.squeeze(-1) * attn_mask
        scores = scores.squeeze(-1) + attn_mask
        # scores = scores.masked_fill(attn_mask==0, -1e9)
        scores = torch.softmax(scores, dim = -1) ## [bs, L, 1]
        output = torch.matmul(scores.unsqueeze(-2), v_s)
        output = output.squeeze(-2)
        return output

class AttentionCate(nn.Module):
    def __init__(self, d_in, d_out):
        super(AttentionCate, self).__init__()
        # 输入进来的QKV是相等的，我们会使用映射linear做一个映射得到参数矩阵Wq, Wk,Wv
        self.W_Q = nn.Linear(d_in, d_out)
        self.W_K = nn.Linear(d_in, d_out)

        self.linear = nn.Linear(d_out, 1)
        self.sigmoid = nn.Sigmoid()
        # self.layer_norm = nn.LayerNorm(d_out)
        # nn.init.xavier_normal_(self.W_Q.weight)
        # nn.init.xavier_normal_(self.W_K.weight)
        # nn.init.xavier_normal_(self.W_V.weight)

    def forward(self, Q, K, V, attn_mask):
        q_s = self.W_Q(Q)  # q_s: [bs, 1,d]   [bs, c,1,d]
        k_s = self.W_K(K)  # k_s: [bs,L,d]    [bs, c, L, d]
        # scores = torch.matmul(q_s, k_s.transpose(-1,-2)) / np.sqrt(q_s.size(-1))
        scores = self.linear(self.sigmoid(q_s + k_s))  # [bs, L, 1]  [bs,c,L,1]
        scores = scores * attn_mask
        # scores = scores.masked_fill(attn_mask==0, -1e9)
        # scores = torch.softmax(scores, dim = 1) ## [bs, L, 1]
        output = torch.matmul(scores.transpose(-1, -2), V)
        output = output.squeeze(-2)
        return output  # [bs,d]  [bs, c,d]




class HGCAN(nn.Module):
    def __init__(self, opt, n_items, num_category, max_length, adj_items, weight_items, adj_cate,
                 weight_cate,adj_c2i,weight_c2i,adj_i2c ,device):
        super(HGCAN, self).__init__()
        self.n_items = n_items
        self.num_category = num_category
        self.hidden_size = opt.hidden_size
        self.max_length  = max_length
        self.device = device
        self.aba = opt.aba
        self.batch_size = opt.batch_size
        self.self_attn1 = SelfAttention1(self.hidden_size, self.hidden_size)
        self.attn = Attention(d_in = self.hidden_size, d_out = self.hidden_size)
        self.attn1 = Attention1(d_in = self.hidden_size, d_out = self.hidden_size)
        self.attn_cate = AttentionCate(d_in=self.hidden_size , d_out = self.hidden_size)


        self.hop = opt.hop
        self.sample_num = opt.sample_num
        self.adj_items = torch.LongTensor(adj_items).to(device)
        self.weight_items = torch.FloatTensor(weight_items).to(device)
        self.adj_cate = torch.LongTensor(adj_cate).to(device)
        self.weight_cate = torch.FloatTensor(weight_cate).to(device)
        self.adj_c2i = torch.LongTensor(adj_c2i).to(device)
        self.weight_c2i = torch.FloatTensor(weight_c2i).to(device)
        self.adj_i2c = torch.LongTensor(adj_i2c).to(device) #[i,1]

        self.i2i_agg = []
        for i in range(self.hop):
            agg = GNN(self.hidden_size, self.device)
            self.add_module('i2i_gnn_{}'.format(i), agg)
            self.i2i_agg.append(agg)

        self.i2c_agg = []
        for i in range(self.hop):
            agg = GNN(self.hidden_size, self.device)
            self.add_module('i2c_gnn_{}'.format(i), agg)
            self.i2c_agg.append(agg)

        self.c2c_agg = []
        for i in range(self.hop):
            agg = GNN(self.hidden_size, self.device)
            self.add_module('c2c_gnn_{}'.format(i), agg)
            self.c2c_agg.append(agg)
        self.c2i_agg = []
        for i in range(self.hop):
            agg = GNN(self.hidden_size, self.device)
            self.add_module('c2i_gnn_{}'.format(i), agg)
            self.c2i_agg.append(agg)

        self.item_embeddings = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
        self.category_embedding = nn.Embedding(self.num_category + 1, self.hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(self.max_length +1, self.hidden_size ,padding_idx=0 )

        self.gate = nn.Sequential(
            nn.Linear(self.hidden_size , self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size , 1)
        )

        self.NN_DivCate_pos1= nn.Linear(2 * self.hidden_size, self.hidden_size, bias=False)
        self.NN_DivCate_pos2 = nn.Linear(2 * self.hidden_size , self.hidden_size, bias=False)
        self.NN_DivCateEncoder = nn.Linear(2 * self.hidden_size , self.hidden_size , bias=False)
        self.NN_IncateEncoder = nn.Linear( 2*self.hidden_size , self.hidden_size , bias=False)
        self.NN_InsessEncoder = nn.Linear(2 * self.hidden_size , self.hidden_size, bias=False)

        self.LN1 = nn.LayerNorm(self.hidden_size)
        self.LN2 = nn.LayerNorm(self.hidden_size)

        self.a1 = nn.Parameter(torch.randn(100,1), requires_grad=True)
        self.a2 = nn.Parameter(torch.randn(100,1), requires_grad=True)

        self.cate_loss_function = nn.CrossEntropyLoss()
        self.item_loss_function = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, inp_sess_padding , inp_sess_cat_padding, lengths, item_cate_matrix, cate_matrix,
                item_cate_mask_inf, cate_mask_inf, query_matrix, reverse_positon_idx, reverse_positon_idx1):


        x = self.item_embeddings.weight[1:]  #[item-1 , d]
        x_nb = self.adj_items[1:]  #[items-1, sample_num ]
        x_weight = self.weight_items[1:]
        y = self.category_embedding.weight[1:-1]
        y_nb = self.adj_cate[1:]
        y_weight = self.weight_cate[1:]
        c2i_nb = self.adj_c2i[1:]
        c2i_weight = self.weight_c2i[1:]
        i2c_nb = self.adj_i2c

        cate_vectors = [y]
        item_vectors = [x]
        for i in range(self.hop):
            #              [i-1 ,d]         [i- 1 , m , d]      [i-1 , m]
            x1 = self.i2i_agg[i](x=item_vectors[i] ,x_nb=item_vectors[i][x_nb - 1] , weight = x_weight) #[items,d]
            #                  [i-1, d]    [i -1 ,1 , d]         [i-1, 1]
            x2 =self.c2i_agg[i](x= item_vectors[i] ,x_nb=cate_vectors[i][i2c_nb - 1]  , weight = torch.ones(x.size(0), 1).to(self.device)) #[items,d]
            #                 [c-1 , d]         [c-1 ,m ,d]     [c-1, m]
            y1 = self.c2c_agg[i](x=cate_vectors[i], x_nb=cate_vectors[i][y_nb - 1] , weight = y_weight ) #[cate ,d]
            #                  [c-1 , d]        [c-1 ,m ,d]        [c-1, m]
            y2 = self.i2c_agg[i](x=cate_vectors[i], x_nb=item_vectors[i][c2i_nb - 1] ,weight  = c2i_weight ) ##[cate ,d]
            y = (y1 +y2)/2 + cate_vectors[i]
            x = (x1 + x2)/2 +item_vectors[i]
            item_vectors.append(x)
            cate_vectors.append(y)
        item_vectors_out = item_vectors[-1]
        cate_vectors_out = cate_vectors[-1]

        item_vectors = self.LN1(item_vectors_out)
        cate_vectors = self.LN2(cate_vectors_out)

        pad_vector = torch.zeros(1, self.hidden_size).to(self.device)
        item_vectors = torch.cat((pad_vector, item_vectors), dim=0)  #[items, d]
        cate_vectors = torch.cat((pad_vector, cate_vectors), dim=0)  # [items, d]
        if self.aba == 1:
            item_vectors = self.item_embeddings.weight
            cate_vectors = self.category_embedding.weight

        inp_emb = item_vectors[inp_sess_padding] # bs * L * d
        mean_f = torch.where(inp_sess_padding ==0, inp_sess_padding,
                                           torch.ones_like(inp_sess_padding).to(self.device))

        all_item_rep = item_vectors[1:]
        all_cate_rep = cate_vectors[1:]

# Category-View Session Representation Learning: hc
        cate_emb = cate_vectors[inp_sess_cat_padding]  # bs * L * d
        pos_emb = self.position_embedding(reverse_positon_idx)  # [bs, L, d]
        con_emb = self.NN_IncateEncoder(torch.cat([cate_emb, pos_emb], dim=-1))  # [bs,L,2d]
        item_q1 = cate_emb[torch.arange(self.batch_size), lengths - 1]
        output_cat = self.attn_cate(item_q1.unsqueeze(1), con_emb, cate_emb, mean_f.unsqueeze(-1))

#Item-View Session Representation Learning:

        # session representations from the whole session: hs
        pos_emb = self.position_embedding(reverse_positon_idx)      # [bs, L, d]
        con_emb = self.NN_InsessEncoder(torch.cat((inp_emb ,pos_emb),dim=-1))
        item_q = inp_emb[torch.arange(self.batch_size), lengths - 1]
        select = self.attn(item_q.unsqueeze(1), con_emb, inp_emb, mean_f.unsqueeze(-1))

        #session representations  from the category subsequences: hz
        bs, c, L = item_cate_matrix.size()
        inp_emb = item_vectors[item_cate_matrix]  # [bs, c， L ,d ]
        cate_emb2 = cate_vectors[cate_matrix]
        cate_emb2 = cate_emb2.unsqueeze(2).expand(bs, c, L, self.hidden_size)  # [bs,c,L,d]
        inp_emb = self.NN_DivCateEncoder(torch.cat([inp_emb, cate_emb2], dim=-1))
        pos_emb1 = self.position_embedding(reverse_positon_idx1)  # [ba, c， L ,d ]
        inp_emb1 = self.NN_DivCate_pos1(torch.cat([inp_emb, pos_emb1], dim=-1))  # [bs,c, L,d]
        attn_output1 = self.self_attn1(inp_emb, inp_emb1, inp_emb,
                                       item_cate_mask_inf.unsqueeze(2).expand(bs, c, L, L))  # [bs,c,L,d]
        rep_cate = attn_output1
        rep_cate1 = self.NN_DivCate_pos2(torch.cat([rep_cate, pos_emb1], dim=-1))
        query = torch.zeros((bs, c, self.hidden_size)).to(self.device)
        split_out = torch.split(rep_cate, 1, dim=1)
        c_i = 0
        for chunk in split_out:  # [bs, 1, L,d]
            q = torch.split(query_matrix, 1, dim=1)[c_i].reshape(1, bs)
            chunk_i = chunk.squeeze(1)[torch.arange(bs), q - 1].transpose(1, 0)  # [1,bs ,d] ->[bs,1 ,d]
            query[torch.arange(bs), c_i] = chunk_i.squeeze(1)
            c_i += 1
        attn_output2 = self.attn1(query.unsqueeze(2), rep_cate1, rep_cate, item_cate_mask_inf)  # .squeeze(2)
        attn_output2 = attn_output2  # +  query
        choose = torch.softmax(self.gate(attn_output2).squeeze(2) + cate_mask_inf, dim=-1)  # [bs，c]
        output2 = torch.sum(choose.unsqueeze(2) * attn_output2, dim=-2)  # [bs ,d]


        Item_output = torch.sigmoid(self.a2) * select + torch.sigmoid(self.a1) * output2
        cate_scores = torch.matmul(output_cat, all_cate_rep.transpose(1, 0))  # [bs, cate-1]


        if self.aba ==4:
            Item_output = select
        if self.aba == 3:
            Item_output = output2

        item_scores = torch.matmul(Item_output, all_item_rep.transpose(1, 0))  # [bs, cate-1]

        return cate_scores, item_scores


    def CE_Loss(self, predict, label):
        logsoftmax_output = torch.log(predict)
        nllloss_func = nn.NLLLoss()
        ce = nllloss_func(logsoftmax_output, label)

        return ce



