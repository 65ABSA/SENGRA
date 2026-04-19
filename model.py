import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import BertModel
from scheme.RGAT import RGATEncoder


class base_model(nn.Module):
    def __init__(self, pretrained_model_path, hidden_dim, dropout,class_n =16, span_average = False, dep_dim=0, dep_n=0):
        super().__init__()
        
        # Encoder
        self.bert = BertModel.from_pretrained(pretrained_model_path)
        self.dense = nn.Linear(self.bert.pooler.dense.out_features, hidden_dim)
        self.span_average = span_average
        self.use_syntax = False if dep_dim == 0 else True
        self.dep_embedding = (
            nn.Embedding(dep_n, dep_dim, padding_idx=0)
            if self.use_syntax
            else None
        )
        # 图神经网络
        self.Graph_encoder = RGATEncoder(
            num_layers=2,
            d_model=hidden_dim,
            heads=4,
            d_ff=768,
            dep_dim=dep_dim,
            att_drop=0.1,
            dropout=0.0,
            use_structure=True
        )
        self.gate_map = nn.Linear(hidden_dim * 2, hidden_dim)  # 门控机制

        # Classifier
        # self.classifier = nn.Linear(hidden_dim * 3, class_n)
        self.classifier = nn.Linear(hidden_dim * 5, class_n)
        
        # dropout
        self.layer_drop = nn.Dropout(dropout)
        
        
    def forward(self, inputs, weight=None):
        
        #############################################################################################
        # word representation
        bert_token = inputs['bert_token']
        attention_mask = (bert_token>0).int()
        bert_word_mapback = inputs['bert_word_mapback']
        token_length = inputs['token_length']
        bert_length = inputs['bert_length']
        deprel_head = inputs['deprel_head']
        deprel = inputs['deprel']
        
        
        bert_out = self.bert(bert_token,attention_mask = attention_mask).last_hidden_state # \hat{h}
        
        bert_seq_indi = sequence_mask(bert_length).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(bert_length) + 1, :] * bert_seq_indi.float()
        word_mapback_one_hot = (F.one_hot(bert_word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)
        
        
        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))
        wnt = word_mapback_one_hot.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        bert_out = bert_out / wnt.unsqueeze(dim=-1)  # h_i

        #############################################################################################
        # 句法表示 (Syntax Representation)
        adj = None
        if self.use_syntax:
            maxlen = max(token_length)  # 获取最大序列长度
            # 生成依存矩阵和标签矩阵
            adj_lst, label_lst = [], []
            for idx in range(len(token_length)):
                adj_i, label_i = head_to_adj(
                    maxlen,
                    deprel_head[idx],
                    deprel[idx],
                    token_length[idx],
                    directed=False,
                    self_loop=True,
                )
                adj_lst.append(adj_i.reshape(1, maxlen, maxlen))
                label_lst.append(label_i.reshape(1, maxlen, maxlen))
            adj = np.concatenate(adj_lst, axis=0)  # 邻接矩阵 [B, maxlen, maxlen]
            adj = torch.from_numpy(adj).cuda()
            labels = np.concatenate(label_lst, axis=0)  # 标签矩阵 [B, maxlen, maxlen]
            label_all = torch.from_numpy(labels).cuda()
            mask = adj.eq(0)
            key_padding_mask = sequence_mask_reverse(token_length)  # 创建掩码
            # dep_relation_embs = self.dep_embedding(label_all)
            dep_relation_embs = None
            graph_out = self.Graph_encoder(
                bert_out, mask=mask, src_key_padding_mask=key_padding_mask, structure=dep_relation_embs
            )
            gate = torch.sigmoid(self.gate_map(torch.cat([graph_out, bert_out], dim=-1)))
            graph_outputs = gate * graph_out + (1 - gate) * bert_out
            merged_outputs = torch.cat([graph_outputs, bert_out], dim=-1)
            # merged_outputs = graph_outputs
        else:
            merged_outputs = bert_out

        #############################################################################################
        # span representation
        
        max_seq = bert_out.shape[1]
        
        token_length_mask = sequence_mask(token_length)
        candidate_tag_mask = torch.triu(torch.ones(max_seq,max_seq,dtype=torch.int64,device=bert_out.device),diagonal=0).unsqueeze(dim=0) * (token_length_mask.unsqueeze(dim=1) * token_length_mask.unsqueeze(dim=-1))
        
        boundary_table_features = torch.cat([merged_outputs.unsqueeze(dim=2).repeat(1,1,max_seq,1), merged_outputs.unsqueeze(dim=1).repeat(1,max_seq,1,1)],dim=-1) * candidate_tag_mask.unsqueeze(dim=-1)  # h_i ; h_j
        span_table_features = form_raw_span_features(bert_out, candidate_tag_mask, is_average = self.span_average) # sum(h_i,h_{i+1},...,h_{j})
        
        # h_i ; h_j ; sum(h_i,h_{i+1},...,h_{j})
        table_features = torch.cat([boundary_table_features, span_table_features],dim=-1)
       
        #############################################################################################
        # classifier
        logits = self.classifier(self.layer_drop(table_features)) * candidate_tag_mask.unsqueeze(dim=-1)
        
        outputs = {
            'logits':logits,
            'adjs':adj
        }
        
        if 'golden_label' in inputs and inputs['golden_label'] is not None:
            loss = calcualte_loss(logits, inputs['golden_label'],candidate_tag_mask, weight = weight)
            outputs['loss'] = loss
        
        return outputs
            
 
def sequence_mask(lengths, max_len=None):

    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) < (lengths.unsqueeze(1))

def sequence_mask_reverse(lengths, max_len=None):

    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))

def form_raw_span_features(v, candidate_tag_mask, is_average = True):
    new_v = v.unsqueeze(dim=1) * candidate_tag_mask.unsqueeze(dim=-1)
    span_features = torch.matmul(new_v.transpose(1,-1).transpose(2,-1), candidate_tag_mask.unsqueeze(dim=1).float()).transpose(2,1).transpose(2,-1)

    
    if is_average:
        """
        _, max_seq, _ = v.shape
        sub_v = torch.tensor(range(1,max_seq+1), device = v.device).unsqueeze(dim=-1)  - torch.tensor(range(max_seq),device = v.device)
        sub_v  = torch.where(sub_v > 0, sub_v, 1).T
        
        span_features = span_features / sub_v.unsqueeze(dim=0).unsqueeze(dim=-1)
        """
        
    return span_features


def head_to_adj(sent_len, head, label, len_, directed=False, self_loop=True):
    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)  # 邻接矩阵
    label_matrix = np.zeros((sent_len, sent_len), dtype=np.int64)  # 标签矩阵

    head = head[:len_]  # 依存关系头索引
    label = label[:len_]  # 依存关系标签序列

    # asp_idx = [idx for idx in range(len(mask)) if mask[idx] == 1]

    for idx, head in enumerate(head):
        if head != 0:
            adj_matrix[idx, head - 1] = 1  # 建立从当前单词到其头节点的边
            label_matrix[idx, head - 1] = label[idx]  # 设置对应的依存关系标签
        else:
            if self_loop:
                adj_matrix[idx, idx] = 1  # 添加自环
                label_matrix[idx, idx] = 2  # 自环标签设为特定值（如 `2`）
                continue  # 自环处理后直接跳过当前节点

        if not directed:
            adj_matrix[head - 1, idx] = 1  # 反向连接
            label_matrix[head - 1, idx] = label[idx]  # 反向边的标签与正向相同

        if self_loop:
            adj_matrix[idx, idx] = 1  # 自环
            label_matrix[idx, idx] = 2  # 自环标签

    return adj_matrix, label_matrix

def calcualte_loss(logits, golden_label,candidate_tag_mask, weight=None):
    loss_func = nn.CrossEntropyLoss(weight = weight, reduction='none')
    return (loss_func(logits.view(-1,logits.shape[-1]), 
                      golden_label.view(-1)
                      ).view(golden_label.size()) * candidate_tag_mask).sum()
    