import datetime
import math

import dgl
import dgl.nn.pytorch as dglnn
from log import logger
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.utils.data import DataLoader
from tqdm import tqdm

from graph.aggregator import LocalAggregator
from graph.collate import gnn_collate_fn, collate
from graph.graph_construction import seq_to_hetero_graph


class MCGNN(nn.Module):

    def __init__(self,
                 args,
                 num_item,
                 num_cat,
                 device,
                 batch_norm=True,
                 feat_drop=0.0,
                 attention_drop=0.0
                 ):
        super(MCGNN, self).__init__()

        self.device = device
        self.batch = args.batch
        self.alpha = args.alpha
        self.embedding_dim = args.emb_size
        self.neighbor_num = args.neighbor_num
        self.graph_feature_select = args.graph_feature_select
        self.item_embedding = nn.Embedding(num_item, self.embedding_dim, max_norm=1)
        self.cate_embedding = nn.Embedding(num_cat, self.embedding_dim, max_norm=1)
        self.pos_embedding = nn.Embedding(200, self.embedding_dim)

        self.dropout30 = nn.Dropout(args.dropout1)
        self.dropout40 = nn.Dropout(args.dropout2)
        self.merge_n_c = nn.Linear(self.embedding_dim, self.embedding_dim, bias=True)

        self.num_layers = args.num_layers  # hyper-parameter for gnn layers
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim * 2) if batch_norm else None

        self.nor = args.norm

        self.finalfeature = FeatureSelect(self.embedding_dim, type=self.graph_feature_select)
        self.gnn_layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.gnn_layers.append(
                dglnn.HeteroGraphConv({
                    'i2i':
                        # dglnn.GATConv(in_channels, hidden_channels // num_heads, num_heads, allow_zero_in_degree=True)
                        dglnn.GATConv(self.embedding_dim, self.embedding_dim, 1, feat_drop=feat_drop, attn_drop=attention_drop,residual=True),
                    'c2c':
                        dglnn.GATConv(self.embedding_dim, self.embedding_dim, 1,feat_drop=feat_drop, attn_drop=attention_drop,residual=True),
                    'c2i':
                        dglnn.GATConv(self.embedding_dim, self.embedding_dim, 1, feat_drop=feat_drop, attn_drop=attention_drop,residual=True),
                    'i2c':
                        dglnn.GATConv(self.embedding_dim, self.embedding_dim, 1, feat_drop=feat_drop, attn_drop=attention_drop,residual=True),
                }, aggregate='sum'))

        # W_h_e * (h_s || e_u) + b
        # self.W_pos = nn.Parameter(th.Tensor(self.embedding_dim * 2 + self.auxemb_dim, self.embedding_dim))
        self.W_pos = nn.Parameter(th.Tensor(self.embedding_dim * 2, self.embedding_dim))
        self.W_hs_e = nn.Parameter(th.Tensor(self.embedding_dim * 2, self.embedding_dim))
        self.W_h_e = nn.Parameter(th.Tensor(self.embedding_dim * 2, self.embedding_dim))
        self.W_c = nn.Parameter(th.Tensor(self.embedding_dim * 2, self.embedding_dim))
        self.feat_drop = nn.Dropout(feat_drop)
        self.fc_sr = nn.Linear(self.embedding_dim * 2, self.embedding_dim, bias=False)
        self.local_agg = LocalAggregator(self.embedding_dim, self.alpha, dropout=0.0)
        self.dropout_local = args.dropout_local

        self.linear_out = nn.Linear(2 * self.embedding_dim, self.embedding_dim, bias=True)
        self.q = nn.Parameter(torch.Tensor(self.embedding_dim, 1))
        self.attn_layernorm = torch.nn.LayerNorm(self.embedding_dim, eps=1e-8)
        self.fwd_layernorm = torch.nn.LayerNorm(self.embedding_dim, eps=1e-8)


        self.w_1 = nn.Parameter(torch.Tensor(2 * self.embedding_dim, self.embedding_dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.embedding_dim, 1))
        self.glu1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.glu2 = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)

        self.x = Parameter(torch.Tensor(1))
        self.y = Parameter(torch.Tensor(1))

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_dc_step, gamma=args.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.embedding_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def handle_local(self, hidden, mask):

        mask = mask.float().unsqueeze(-1)

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        # b = self.embedding.weight[1:]  # n_nodes x latent_size
        # scores = torch.matmul(select, b.transpose(1, 0))
        return select

    def compute_scores(self, hidden, mask):
            mask = mask.float().unsqueeze(-1)  # b* one * d

            batch_size = hidden.shape[0]
            len = hidden.shape[1]
            pos_emb = self.pos_embedding.weight[:len]
            pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

            hs = torch.matmul(hidden, self.q)
            beta_a = F.softmax(hs, dim=1)
            beta_a = beta_a * mask
            hs = torch.sum(beta_a * hidden, 1)
            hs = self.attn_layernorm(hs)
            #        hs = hs.unsqueeze(-2).repeat(1, len, 1)  #b*l*d
            #        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
            hh = hidden[torch.arange(mask.shape[0]).long(), 0]
            #        hh = hh.unsqueeze(-2).repeat(1, len, 1)
            #        hs = hh + hs * self.aaa
            hs = self.linear_out(torch.cat([hs, hh], 1))
            hs = hs.unsqueeze(-2).repeat(1, len, 1)  # b*l*d

            #        hs = torch.transpose(hs, 0, 1)
            #        hidden = torch.transpose(hidden, 0, 1)
            #        mha_outputs, attn_output_weights = self.attn_layer(hs, hidden, hidden)
            #        hs = hs + mha_outputs * self.aaa
            #        hs = torch.transpose(hs, 0, 1)
            #        hs = self.fwd_layernorm(hs).squeeze()
            nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
            nh = torch.tanh(nh)
            nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
            beta = torch.matmul(nh, self.w_2)
            beta = beta * mask
            select = torch.sum(beta * hidden, 1)
            #   select = self.fwd_layernorm(select)
            if self.nor == True:
                select = self.fwd_layernorm(select)
            # b = self.embedding.weight[1:]  # n_nodes x latent_size
            # scores = torch.matmul(select, b.transpose(1, 0))
            return select

    def compute_scores_npos(self, hidden, mask):
        mask = mask.float().unsqueeze(-1)  # b* one * d

        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        # pos_emb = self.pos_embedding.weight[:len]
        # pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)

        hs = torch.matmul(hidden, self.q)
        beta_a = F.softmax(hs, dim=1)
        beta_a = beta_a * mask
        hs = torch.sum(beta_a * hidden, 1)
        hs = self.attn_layernorm(hs)
        #        hs = hs.unsqueeze(-2).repeat(1, len, 1)  #b*l*d
        #        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hh = hidden[torch.arange(mask.shape[0]).long(), 0]
        #        hh = hh.unsqueeze(-2).repeat(1, len, 1)
        #        hs = hh + hs * self.aaa
        hs = self.linear_out(torch.cat([hs, hh], 1))
        hs = hs.unsqueeze(-2).repeat(1, len, 1)  # b*l*d

        #        hs = torch.transpose(hs, 0, 1)
        #        hidden = torch.transpose(hidden, 0, 1)
        #        mha_outputs, attn_output_weights = self.attn_layer(hs, hidden, hidden)
        #        hs = hs + mha_outputs * self.aaa
        #        hs = torch.transpose(hs, 0, 1)
        #        hs = self.fwd_layernorm(hs).squeeze()
        nh = hidden
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)
        #   select = self.fwd_layernorm(select)
        if self.nor == True:
            select = self.fwd_layernorm(select)
        # b = self.embedding.weight[1:]  # n_nodes x latent_size
        # scores = torch.matmul(select, b.transpose(1, 0))
        return select




    def feature_encoder(self, g: dgl.DGLHeteroGraph):
        iid = g.nodes['i'].data['id']
        cid = g.nodes['c'].data['id']

        # store the embedding in graph
        # g.update_all(fn.copy_e('pos', 'ft'),
        #              fn.min('ft', 'f_pos'),
        #              etype='c2i')
        # pos_emb = self.pos_embedding(g.nodes['i'].data['f_pos'].long())
        # cat_emb = th.cat([
        #     self.item_embedding(iid), pos_emb,
        #     self.cate_embedding(g.nodes['i'].data['cate'])
        # ],
        #     dim=1)
        cat_emb = th.cat([
            self.item_embedding(iid),
            self.cate_embedding(g.nodes['i'].data['cate'])
        ],
            dim=1)
        g.nodes['i'].data['f'] = th.matmul(cat_emb, self.W_pos)
        g.nodes['c'].data['f'] = self.cate_embedding(cid)

    def forward(self, g: dgl.DGLHeteroGraph, seq_len, seq_len_uniq, u_input, category_ids, alias_inputs, adj,
                mask_item):

        h_emb = self.item_embedding(u_input)
        # local
        h_local = self.local_agg(h_emb, adj, mask_item)
        h_local = F.dropout(h_local, self.dropout_local, training=self.training)
        get = lambda index: h_local[index][alias_inputs[index]]
        seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])


        self.feature_encoder(g)

        h = [{
            'i': g.nodes['i'].data['f'],
            'c': g.nodes['c'].data['f']
        }]
        for i, layer in enumerate(self.gnn_layers):
            out = layer(g, (h[-1], h[-1]))
            h.append(out)

        last_nodes = g.filter_nodes(lambda nodes: nodes.data['last'] == 1, ntype='i')  # index array
        last_cnodes = g.filter_nodes(lambda nodes: nodes.data['clast'] == 1, ntype='c')

        # try gated feat
        feat = self.finalfeature(h)

        h_i = feat['i'][last_nodes].squeeze()  # [bs, embsize]

        # h_sum = feat['i'][:seq_len_uniq[0]].sum(0) / seq_len_uniq[0]
        # h_sum = h_sum.unsqueeze(0)  # 1*embsize

        h_cc = F.dropout(feat['i'], 0.2, training=self.training)
        temp = feat['i'].split(seq_len_uniq.tolist(),dim = 0)
        get2 = lambda index: temp[index][alias_inputs[index][:seq_len_uniq.tolist()[index]]-1]
        h_he = torch.stack([torch.cat([get2(i), torch.zeros(u_input.shape[1]-seq_len_uniq.tolist()[i],self.embedding_dim)], dim=0)
                                   for i in torch.arange(len(alias_inputs)).long()])


        # h_he = th.zeros(u_input.shape[0],u_input.shape[1],self.embedding_dim)
        # for j in range(len(seq_len_uniq)):
        #     x = torch.Tensor().to(self.device)
        #     start = 0 if j == 0 else seq_len_uniq[j - 1]
        #     x_uniqu = feat['i'][start : seq_len_uniq[j]]
        #     for k in alias_inputs[j][: seq_len[j]]:
        #         x = th.cat((x,x_uniqu[k-1].unsqueeze(0)),dim = 0)
        #
        #     # torch.sum(mask_item.float(), -1).repeat(1, , ).size()
        #     h_he[j][:seq_len[j]] = x.view(-1, self.embedding_dim)
        #
        # h_he = h_he.to(self.device)

        h_all = seq_hidden*self.x + h_he

        gate = th.sigmoid(self.W_hs_e(th.cat([seq_hidden, h_he], dim=-1)))
        ifeature = gate * seq_hidden + (1 - gate) * h_he

        # h_l = self.handle_local(h_all, mask_item)
        h_l = self.compute_scores(h_all, mask_item)

        # gate = th.sigmoid(th.matmul(th.cat((h_l, h_i), 1), self.W_hs_e))
        # h_all = gate *  + (1 - gate) * h_sum + h_l  # [bs, embsize]
        # h_all = h_l  + h_i
        h_all = h_l  + h_i*self.y

        # ____________________neiighbor_______________________________
        sess_current = h_all

        # cosine similarity
        fenzi = torch.matmul(sess_current, sess_current.permute(1, 0))  # 512*512
        fenmu_l = torch.sum(sess_current * sess_current + 0.000001, 1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))
        cos_sim = fenzi / fenmu  # 512*512
        cos_sim = nn.Softmax(dim=-1)(cos_sim)  # zz 5128'512x

        k_v = self.neighbor_num
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1)
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = sess_current[topk_indice]
        # logger.error("sess_topk size: %s", sess_topk.size())

        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.embedding_dim)
        # logger.error("cos_sim size: %s", cos_sim.size())

        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        neighbor_sess = self.dropout40(neighbor_sess)

        sess_current0 = sess_current
        neighbor_sess0 = neighbor_sess

        gamma = ((neighbor_sess0 * sess_current0).sum(dim=1) / (
                torch.norm(neighbor_sess0, p=2, dim=1) * torch.norm(sess_current0, p=2, dim=1))).unsqueeze(1)
        sess_final = sess_current + gamma * neighbor_sess

        # sess_final = torch.cat(
        #     [sess_current, neighbor_sess, sess_current + neighbor_sess, sess_current * neighbor_sess], 1)

        sess_final = self.dropout30(sess_final)
        sess_final = self.merge_n_c(sess_final)

        # ____________________________________________________________

        feat_last_cate = feat['c'][last_cnodes].squeeze()

        item_embeddings = self.item_embedding.weight[1:]
        item_score = th.matmul(sess_final, item_embeddings.t())

        cate_embeddings = self.cate_embedding.weight[1:]
        cate_score = th.matmul(feat_last_cate, cate_embeddings.t())

        return item_score, cate_score, g.batch_num_nodes('i')


class FeatureSelect(nn.Module):
    def __init__(self, embedding_dim, type='last'):
        super().__init__()
        self.embedding_dim = embedding_dim
        assert type in ['last', 'mean', 'gated']
        self.type = type

        self.W_g1 = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.W_g2 = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.W_g3 = nn.Linear(2 * self.embedding_dim, self.embedding_dim)

    def forward(self, h):
        h[0]['i'] = h[0]['i'].squeeze()
        h[-1]['i'] = h[-1]['i'].squeeze()
        h[0]['c'] = h[0]['c'].squeeze()
        h[-1]['c'] = h[-1]['c'].squeeze()
        feature = None
        if self.type == 'last':
            feature = h[-1]
        elif self.type == 'gated':
            gate = th.sigmoid(self.W_g1(th.cat([h[0]['i'], h[-1]['i']], dim=-1)))
            ifeature = gate * h[0]['i'] + (1 - gate) * h[-1]['i']

            gate = th.sigmoid(self.W_g3(th.cat([h[0]['c'], h[-1]['c']], dim=-1)))
            cfeature = gate * h[0]['c'] + (1 - gate) * h[-1]['c']

            feature = {'i': ifeature, 'c': cfeature}

        elif self.type == 'mean':
            isum = th.zeros_like(h[0]['i'])
            csum = th.zeros_like(h[0]['c'])
            for data in h:
                isum += data['i'].view(-1,self.embedding_dim)
                csum += data['c'].view(-1,self.embedding_dim)
            feature = {'i': isum / len(h), 'c': csum / len(h)}

        return feature


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable


def forward(model, data):
    bgs, label, next_cate, seq_len, seq_len_uniq, u_input, alias_inputs, category_ids, adj, mask = data


    u_input = u_input.to(model.device)
    alias_inputs = alias_inputs.to(model.device)
    category_ids = category_ids.to(model.device)
    adj = adj.to(model.device)
    mask = mask.to(model.device)
    seq_len = seq_len.to(model.device)
    seq_len_uniq = seq_len_uniq.to(model.device)
    bgs = bgs.to(model.device)

    item_score, cate_score, session_length = model.forward(bgs, seq_len, seq_len_uniq, u_input, category_ids,
                                                           alias_inputs, adj, mask)
    # item_score = item_score * mask
    # cate_score = cate_score * mask
    return item_score, cate_score, label, next_cate

# collate_fn = gnn_collate_fn(seq_to_hetero_graph)
def train_test(model, train_loader, test_loader):
    print('start training: ', datetime.datetime.now())
    logger.error('start training: %s ', datetime.datetime.now())

    model.train()
    total_loss = 0.0

    for data in tqdm(train_loader):
        model.optimizer.zero_grad()
        item_score, cate_score, label, next_cate = forward(model, data)
        label = label.to(model.device)
        next_cate = next_cate.to(model.device)

        loss_item = model.loss_function(item_score, label - 1)
        loss_cate = model.loss_function(cate_score, next_cate - 1)
        # print('loss_item:', loss_item)
        # print('loss_cate:', loss_cate)
        # loss_item.backward()
        # loss_cate.backward()
        loss = (loss_item + loss_cate)
        loss.backward()
        model.optimizer.step()

        total_loss = total_loss + loss

    print('\tLoss:\t%.3f' % total_loss)
    logger.error('\tLoss:\t%.3f' % total_loss)
    model.scheduler.step()

    print('start predicting: ', datetime.datetime.now())
    logger.error('start predicting: %s ', datetime.datetime.now())
    model.eval()

    result = []
    hit_k10, mrr_k10, hit_k20, mrr_k20, hit_k30, mrr_k30, hit_k40, mrr_k40, hit_k50, mrr_k50 = [], [], [], [], [], [], [], [], [], []


    for data in test_loader:
        scores, cate_score, targets, next_cate = forward(model, data)
        sub_scores_k20 = scores.topk(20)[1]
        sub_scores_k20 = trans_to_cpu(sub_scores_k20).detach().numpy()
        sub_scores_k10 = scores.topk(10)[1]
        sub_scores_k10 = trans_to_cpu(sub_scores_k10).detach().numpy()

        sub_scores_k30 = scores.topk(30)[1]
        sub_scores_k30 = trans_to_cpu(sub_scores_k30).detach().numpy()
        sub_scores_k40 = scores.topk(40)[1]
        sub_scores_k40 = trans_to_cpu(sub_scores_k40).detach().numpy()
        sub_scores_k50 = scores.topk(50)[1]
        sub_scores_k50 = trans_to_cpu(sub_scores_k50).detach().numpy()
        targets = targets.numpy()

        for score, target, mask in zip(sub_scores_k20, targets, torch.tensor(test_loader.dataset.mask)):
            hit_k20.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k20.append(0)
            else:
                mrr_k20.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k10, targets, torch.tensor(test_loader.dataset.mask)):
            hit_k10.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k10.append(0)
            else:
                mrr_k10.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k30, targets, torch.tensor(test_loader.dataset.mask)):
            hit_k30.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k30.append(0)
            else:
                mrr_k30.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k40, targets, torch.tensor(test_loader.dataset.mask)):
            hit_k40.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k40.append(0)
            else:
                mrr_k40.append(1 / (np.where(score == target - 1)[0][0] + 1))

        for score, target, mask in zip(sub_scores_k50, targets, torch.tensor(test_loader.dataset.mask)):
            hit_k50.append(np.isin(target - 1, score))
            if len(np.where(score == target - 1)[0]) == 0:
                mrr_k50.append(0)
            else:
                mrr_k50.append(1 / (np.where(score == target - 1)[0][0] + 1))

    result.append(np.mean(hit_k10) * 100)
    result.append(np.mean(mrr_k10) * 100)
    result.append(np.mean(hit_k20) * 100)
    result.append(np.mean(mrr_k20) * 100)

    result.append(np.mean(hit_k30) * 100)
    result.append(np.mean(mrr_k30) * 100)
    result.append(np.mean(hit_k40) * 100)
    result.append(np.mean(mrr_k40) * 100)
    result.append(np.mean(hit_k50) * 100)
    result.append(np.mean(mrr_k50) * 100)

    return result
