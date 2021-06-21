import torch, torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.utils.data
from prettytable import PrettyTable
class MBN_Model(nn.Module):
    def __init__(self, params):
        super(self.__class__, self).__init__()
        self.mix_mode = params.mix_mode
        self.target_task = params.target_task
        self.n_tasks = params.n_tasks
        self.have_tasks = params.have_tasks
        self.repeat = params.repeat
        self.n_items = params.n_items
        self.d_rnn = params.d_rnn
        self.d_item = params.d_item
        self.d_attn = params.d_attn

        self.pool_mode = params.pool_mode
        self.rnn_mode = params.rnn_mode
        self.temp_mode = params.temp_mode
        self.cross_mode = params.cross_mode

        self.meta_rnn = params.meta_rnn
        self.dropout = params.dropout
        self.use_drop = params.use_drop
        self.behavior_type = params.behavior_type
        self.use_residual = params.residual

        self.device = torch.device("cuda:%d" % params.cuda if torch.cuda.is_available() else "cpu")

        self.item_emb = nn.Embedding(params.n_items, params.d_item, padding_idx=0)
        self.d_rnn_input = self.d_item + self.d_rnn if self.cross_mode else self.d_item
        if self.rnn_mode == "rnn":
            self.rnncell = [nn.RNNCell(input_size=self.d_rnn_input, hidden_size=self.d_rnn).to(self.device) for _ in
                            range(self.n_tasks)]
            self.metacell = nn.RNNCell(input_size=self.d_rnn * self.n_tasks, hidden_size=self.d_rnn).to(self.device)
        elif self.rnn_mode == "gru":
            self.grucell = nn.ModuleList([nn.GRUCell(input_size=self.d_rnn_input, hidden_size=self.d_rnn).to(self.device) for _ in
                            range(self.n_tasks)])
            self.metacell = nn.GRUCell(input_size=self.d_rnn * self.n_tasks, hidden_size=self.d_rnn).to(self.device)
        elif self.rnn_mode == "lstm":
            self.lstmcell = [nn.LSTMCell(input_size=self.d_rnn_input, hidden_size=self.d_rnn).to(self.device) for _ in
                             range(self.n_tasks)]
            self.metacell = nn.LSTMCell(input_size=self.d_rnn * self.n_tasks, hidden_size=self.d_rnn).to(self.device)
        else:
            raise NotImplementedError("rnn_mode", self.rnn_mode)
        self.basket_attn = None
        if self.pool_mode == "attn":
            self.basket_attn = AttnPool(self.d_item, self.d_rnn, self.d_attn).to(self.device)
        self.temporal_attn = None
        if self.temp_mode:
            self.temporal_attn = [AttnPool(self.d_rnn, self.d_rnn, self.d_attn).to(self.device) for _ in range(self.n_tasks)]
        if self.repeat in [0, 2]:
            self.new_linear = nn.Linear(self.d_rnn, self.n_items, bias=True)
        if self.repeat in [0, 1]:
            self.copy_linear = [nn.Linear(self.d_rnn, self.n_items, bias=True).to(self.device) for _ in range(self.n_tasks)]
        
        if not self.meta_rnn:
            self.no_meta_linear = nn.Linear(self.d_rnn * self.n_tasks, self.n_items, bias=True)
        if self.use_drop:
            self.dropout_emb = nn.Dropout(self.dropout)
            self.dropout_hidden = nn.Dropout(self.dropout)
        if self.behavior_type:
            self.behavior_attn = AttnPool(self.d_rnn, self.d_rnn, self.d_attn)

    def forward(self, user_vectors, item_vectors, copy_vectors):
        """
        :param user_vectors: [batch_size]
        :param item_vectors: [n_tasks, batch_size, n_steps, basket_size]
        :param copy_vectors: [n_tasks, batch_size, n_steps, copy_size]
        :return:
        """
        batch_size = user_vectors.size()[0]
        
        item_vectors = item_vectors
        sequence_size = item_vectors.size()[2]
        
        # basket pooling
        e_items = self.item_emb(item_vectors)
        n_basket_items = torch.sum((item_vectors > 0).type_as(e_items), -1)
        cell_mask = (n_basket_items[:, :, :] > 0).float()

        if self.pool_mode == "max":
            items_mask = (item_vectors == 0).type_as(e_items)
            e_baskets = torch.max(e_items + items_mask[:, :, :, :, None] * torch.min(e_items), -2)[0]

        elif self.pool_mode == "ave":
            e_baskets_den = n_basket_items[:, :] + (n_basket_items == 0).type_as(n_basket_items)
            e_baskets = torch.sum(e_items, -2) / e_baskets_den[:, :, :, None].type_as(e_items)
        elif self.pool_mode == "attn":
            pass
        elif self.pool_mode == "sum":
            e_baskets = torch.sum(e_items, -1)
        else:
            raise NotImplementedError()

        # recurrent neural network
        h_tasks = [torch.zeros(batch_size, self.d_rnn).to(self.device) for _ in range(self.n_tasks)]
        c_tasks = [torch.zeros(batch_size, self.d_rnn).to(self.device) for _ in range(self.n_tasks)]
        h_t_tasks = [h_tasks[_t].unsqueeze(0) for _t in range(self.n_tasks)]
        h_t_tasks_temp = None
        if self.temp_mode:
            h_t_tasks_temp = [h_tasks[_t].unsqueeze(0) for _t in range(self.n_tasks)]
        if not self.meta_rnn:
            h_not_meta = []
        if self.use_residual:
            h_tasks_residual = [torch.zeros(batch_size, self.d_rnn).to(self.device) for _ in range(self.n_tasks)]

        h_meta = torch.zeros(batch_size, self.d_rnn).to(self.device)
        c_meta = torch.zeros(batch_size, self.d_rnn).to(self.device)
        h_t_meta = h_meta.unsqueeze(0)
        for i in range(sequence_size):
            # tasks rnn
            for ti in range(self.n_tasks):
                h = h_tasks[ti]
                # basket pool
                if self.pool_mode != "attn":
                    cell_input = e_baskets[ti, :, i, :]
                else:
                    cell_input = self.basket_attn(e_items[ti, :, i, :, :], h, item_vectors[ti, :, i, :])

                # meta preprocess
                if self.cross_mode and self.meta_rnn:
                    tmp_cell_input = cell_input
                    cell_input = torch.cat([cell_input, h_meta], dim=-1)

                if self.use_drop:
                    cell_input = self.dropout_emb(cell_input)

                if self.rnn_mode == "rnn":
                    nh = self.rnncell[ti](cell_input, h)
                elif self.rnn_mode == "gru":
                    nh = self.grucell[ti](cell_input, h)
                elif self.rnn_mode == "lstm":
                    c = c_tasks[ti]
                    nh, nc = self.lstmcell[ti](cell_input, (h, c))
                    c_tasks[ti] = nc * cell_mask[ti, :, i, None] + c * (1-cell_mask[ti, :, i, None])

                h = nh * cell_mask[ti, :, i, None] + h * (1 - cell_mask[ti, :, i, None])

                if self.use_drop:
                    h = self.dropout_hidden(h)
                h_t_tasks[ti] = torch.cat([h_t_tasks[ti], h.unsqueeze(0)], dim=0)
                h_tasks[ti] = h
                if self.use_residual:
                    if self.cross_mode and self.meta_rnn:
                        h_tasks_residual[ti] = h + tmp_cell_input * cell_mask[ti, :, i, None]
                    else:
                        h_tasks_residual[ti] = h + cell_input * cell_mask[ti, :, i, None]

            if self.meta_rnn:
                # meta
                if self.use_residual:
                    meta_input = torch.cat(h_tasks_residual, dim=-1)
                else:
                    meta_input = torch.cat(h_tasks, dim=-1)
                if self.rnn_mode == "lstm":
                    h_meta, c_meta = self.metacell(meta_input, (h_meta, c_meta))
                else:
                    h_meta = self.metacell(meta_input, h_meta)
                h_t_meta = torch.cat([h_t_meta, h_meta.unsqueeze(0)], dim=0)
                if self.temp_mode:
                    for ti in range(self.n_tasks):
                        h_late = self.temporal_attn[ti](
                            h_t_tasks[ti][1:].transpose(0, 1), h_meta, n_basket_items[ti, :, :i + 1])
                        h_t_tasks_temp[ti] = torch.cat([h_t_tasks_temp[ti], h_late.unsqueeze(0)], 0)
            else:
                h_not_meta.append(torch.cat(h_tasks, dim=-1))

        if self.temp_mode:
            e_rnn_tasks = [h_t[1:].transpose(0, 1) for h_t in h_t_tasks_temp]
        else:
            e_rnn_tasks = [h_t[1:].transpose(0, 1) for h_t in h_t_tasks]

        if not self.meta_rnn:
            pass
        else:
            e_rnn_meta = h_t_meta[1:].transpose(0, 1)

        if self.repeat == 0: # complete
            if self.meta_rnn:
                score_new_all = self.new_linear(e_rnn_meta)
            else:
                score_new_all = self.no_meta_linear(torch.stack(h_not_meta, 0).transpose(0, 1))
            score_copys = []
            all_zeros = torch.zeros_like(score_new_all)
            for ti in range(self.n_tasks):
                zeros = torch.zeros_like(score_new_all).scatter_(-1, copy_vectors[ti], 1)
                all_zeros += zeros
                score_copy = self.copy_linear[ti](e_rnn_tasks[ti]) * zeros
                score_copys.append(score_copy.unsqueeze(0))
            if self.mix_mode == 1:
                score_copy_tasks = torch.max(torch.cat(score_copys, 0), 0)[0]
            else:
                score_copy_tasks = torch.sum(torch.cat(score_copys, 0), 0)
            score_new = score_new_all * (all_zeros == 0).float()
            score_all_unlog = score_copy_tasks + score_new
            score_copy_tasks = score_copy_tasks + -1000.0*(all_zeros == 0).float()
            score_all = F.log_softmax(score_all_unlog, dim=-1)
        elif self.repeat == 1:  # only repeat
            score_new_all = torch.zeros(batch_size, sequence_size, self.n_items).to(self.device)
            score_copys = []
            all_zeros = torch.zeros_like(score_new_all)
            for ti in range(self.n_tasks):
                zeros = torch.zeros_like(score_new_all).scatter_(-1, copy_vectors[ti], 1)
                all_zeros += zeros
                score_copy = self.copy_linear[ti](e_rnn_tasks[ti]) * zeros
                score_copys.append(score_copy.unsqueeze(0))
            if self.mix_mode == 1:
                score_copy_tasks = torch.max(torch.cat(score_copys, 0), 0)[0]
            else:
                score_copy_tasks = torch.sum(torch.cat(score_copys, 0), 0)
            score_new = score_new_all
            score_all_unlog = score_copy_tasks
            score_all = F.log_softmax(score_all_unlog, dim=-1)
            score_copy_tasks = score_copy_tasks + -1000.0 * (all_zeros == 0).float()
        else:
            if self.meta_rnn:
                score_new_all = self.new_linear(e_rnn_meta)
            else:
                score_new_all = self.no_meta_linear(torch.stack(h_not_meta, 0).transpose(0, 1))
            score_new = score_new_all
            score_copy_tasks = score_new_all
            score_all_unlog = score_new_all
            score_all = F.log_softmax(score_all_unlog, dim=-1)
            score_copys = []
        return score_all, score_copy_tasks, score_new, score_copys

    def layer_wise_parameters(self):
        '''
        the named_parameters() method does not look for all objects that are contained in your model, just the nn.Modules and nn.Parameters, so as I stated above, if you store you parameters outsite of these, then they won’t be detected by named_parameters().
        We can use nn.ModuleList() instead of Python List().
        '''

        table = PrettyTable()
        table.field_names = ["Layer Name", "Output Shape", "Param #"]
        table.align["Layer Name"] = "l"
        table.align["Output Shape"] = "r"
        table.align["Param #"] = "r"
        for name, parameters in self.named_parameters():
            if parameters.requires_grad:
                table.add_row([name, str(list(parameters.shape)), parameters.numel()])
            else:
                print('No grad:')
                print(name)
                print(str(list(parameters.shape)))
                print(parameters.numel())
                print('-'*80)
        return table

class SelfAttn(nn.Module):
    def __init__(self, model_dim, head_dim, n_head):
        self.linear_k = nn.Linear(self.model_dim, self.head_dim * self.n_head) 
        self.linear_v = nn.Linear(self.model_dim, self.head_dim * self.n_head) 
        self.linear_q = nn.Linear(self.model_dim, self.head_dim * self.n_head)

    def forward(self, e_key, e_query, mask_key):
        pass


class AttnPool(nn.Module):
    def __init__(self, d_key, d_query, d_attn):
        super(self.__class__, self).__init__()
        self.attn_key_linear = nn.Linear(d_key, d_attn, bias=True)
        self.attn_query_linear = nn.Linear(d_query, d_attn, bias=False)
        self.attn_outer_linear = nn.Linear(d_attn, 1, bias=False)

    def forward(self, e_key, e_query, mask_key = None):
        """
        :param e_key, shape=[batch_size, key_size, d_key]
        :param e_query, shape=[batch_size, d_query]
        :param mask_key, shape=[batch_size, key_size]
        :return e_attn, shape=[batch_size, d_key]
         """
        key_tmp = self.attn_key_linear(e_key)
        query_tmp = self.attn_query_linear(e_query)
        tmp2 = torch.tanh(key_tmp + query_tmp[:, None, :])
        attn = self.attn_outer_linear(tmp2)[:, :, 0]
        attn_exp = torch.exp(attn)
        if mask_key is not None:
            attn_exp = attn_exp * (mask_key > 0).type_as(attn)
        attn_exp_den = torch.sum(attn_exp, 1)
        attn_exp_den = attn_exp_den + (attn_exp_den == 0).type_as(attn_exp_den)
        attn_softmax = attn_exp / attn_exp_den[:, None]
        e_attn = torch.sum(e_key * attn_softmax[:, :, None], 1)
        return e_attn

def model_loss(logp_seq, item_batch_ix, mask_batch_ix):
    actual_next_tokens = item_batch_ix
    actual_next_num = torch.sum((item_batch_ix > 0).type_as(mask_batch_ix), 2)
    actual_next_num = actual_next_num + (actual_next_num == 0).type_as(actual_next_num)
    predictions_logp = logp_seq * mask_batch_ix[:, :, None] / actual_next_num
    logp_next = torch.gather(predictions_logp, dim=2, index=actual_next_tokens[:, :, :])
    logp_next = logp_next * (actual_next_tokens > 0).type_as(logp_next)
    loss = -logp_next.sum() / mask_batch_ix.sum()
    return loss