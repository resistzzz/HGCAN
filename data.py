
import numpy as np
import copy
import time

def transfer2idx(data, item2idx):
    seqs, labs = data[0], data[1]
    #data[0] = [[1, 2], [1], [4], [6]...]
    #data[1] = [3, 2, 5, 7...]
    for i in range(len(seqs)):
        data[0][i] = [item2idx[s] for s in data[0][i]]
        data[1][i] = item2idx[data[1][i]]
    return data

# generate adjacency matrix
def handle_adj(adj_items, weight_items, n_items, sample_num):
    adj_entity = np.zeros((n_items, sample_num), dtype=np.int)
    wei_entity = np.zeros((n_items, sample_num))
    for entity in range(1, n_items):
        neighbor = list(adj_items[entity])
        neighbor_weight = list(weight_items[entity])
        n_neighbor = len(neighbor)
        if n_neighbor == 0:
            continue
        if n_neighbor >= sample_num:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbor)), size=sample_num, replace=True)
        tmp, tmp_wei = [], []
        for i in sampled_indices:
            tmp.append(neighbor[i])
            tmp_wei.append(neighbor_weight[i])

        adj_entity[entity] = np.array([neighbor[i] for i in sampled_indices])
        wei_entity[entity] = np.array([neighbor_weight[i] for i in sampled_indices])
    return adj_entity, wei_entity


class Data(object):
    '''padding with the maximum length in each batch'''
    def __init__(self, data, n_items):
        self.data = data
        self.n_items = n_items
        self.raw_sessions = np.asarray(data[0])
        self.raw_labs = np.asarray(data[1])
        self.length = len(self.raw_sessions)

    def __len__(self):
        return self.length

    def generate_batch(self, batch_size):
        n_batch = self.length // batch_size
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = np.arange(self.length - batch_size, self.length)
        return slices

    def get_slice_sess_mask(self, index):
        inp_sess = self.raw_sessions[index]
        targets = self.raw_labs[index]
        lengths = []
        for session in inp_sess:
            lengths.append(len(session))
        max_length = max(lengths)
        inp_sess_padding, mask_1, mask_inf = self.zero_padding_mask(inp_sess, max_length)
        return inp_sess, inp_sess_padding, mask_inf, targets, lengths


    def zero_padding_mask(self, data, max_length):
        # Return a mask of the session matrix with padding
        # mask_1 Fill in 1 where there are items in the session
        # mask_inf is 0 where the session has an item and -inf where there is no item
        out_data = np.zeros((len(data), max_length), dtype=np.int)
        mask_1 = np.zeros((len(data), max_length), dtype=np.int)
        mask_inf = np.full((len(data), max_length), float('-inf'), dtype=np.float32)
        for i in range(len(data)):
            out_data[i, :len(data[i])] = data[i]
            mask_1[i, :len(data[i])] = 1
            mask_inf[i, :len(data[i])] = 0.0
        return out_data, mask_1, mask_inf


    def get_item_padding_by_cate(self,batch_size, inp_sess, inp_sess_cat):
        #
        # generate item_cate_matrix ,reflect the correspondence between items and categories
        # cate_matrix includes number of items in the category

        cate_num, cate_item_num, cate_list, cate_item_list = self.get_cate_num(inp_sess_cat)
        item_cate_matrix = np.zeros((batch_size, cate_num, cate_item_num))
        reverse_positon_idx1 = np.zeros((batch_size, cate_num, cate_item_num))
        cate_matrix = np.zeros((batch_size, cate_num))
        query_matrix = np.zeros((batch_size, cate_num))
        item_cate_mask_inf = np.zeros((batch_size, cate_num, cate_item_num), dtype=np.float32)
        cate_mask_inf = np.full((batch_size, cate_num), float('-inf'), dtype=np.float32)
        batch_count = 0
        for sess, sess_cate in zip(inp_sess, inp_sess_cat):
            cate_list = [0] * cate_num
            cate_item_dict = {}
            cate_count = 0
            for i in range(len(sess_cate)):
                if sess_cate[i] in cate_item_dict:
                    cate_item_dict[sess_cate[i]] += [sess[i]]
                else:
                    cate_item_dict[sess_cate[i]] = [sess[i]]
                    cate_list[cate_count] = sess_cate[i]
                    cate_count += 1
            count = 0
            for i in cate_item_dict:
                item_cate_matrix[batch_count][count] = cate_item_dict[i] + [0] *(cate_item_num - len(cate_item_dict[i]))
                reverse_positon_idx1[batch_count][count] = list(np.arange(len(cate_item_dict[i])-1,-1,-1)+1) + [0] * (cate_item_num - len(cate_item_dict[i]))
                query_matrix[batch_count][count] = len(cate_item_dict[i])
                item_cate_mask_inf[batch_count][count][len(cate_item_dict[i]):] = float('-inf')
                count += 1
            cate_matrix[batch_count] = cate_list
            cate_mask_inf[batch_count][:cate_count] = 0
            batch_count += 1

        return item_cate_matrix, cate_matrix, item_cate_mask_inf, cate_mask_inf, query_matrix, reverse_positon_idx1





    def get_cate_num(self, inp_sess_cat):
        # generate cate_list，record how many categories of items appear in each session
        # cate——item——list ， record maximum number of items in a single category per session
        cate_list = []
        cate_item_list =[]
        for seq in inp_sess_cat:

            appear_times = {}
            for lable in seq:
                if lable in appear_times:
                    appear_times[lable] += 1
                else:
                    appear_times[lable] = 1
            most_common = appear_times[max(appear_times, key=lambda x: appear_times[x])]
            cate_item_list = np.append(cate_item_list, most_common)
            cate_list = np.append(cate_list, len(appear_times))
        cate_num = max(cate_list)
        cate_item_num = max(cate_item_list)
        return int(cate_num) , int(cate_item_num) ,cate_list, cate_item_list


    def get_reverse_position(self,inp_sess_padding  , length):
        batch_size, L = np.shape(inp_sess_padding)
        reverse_positon_idx = np.zeros_like(inp_sess_padding, dtype=np.int)
        for i in range(batch_size):
            reverse_positon_idx[i] = list(np.arange(length[i]-1 ,-1,-1)+1) + [0] *(L - length[i])
        return reverse_positon_idx