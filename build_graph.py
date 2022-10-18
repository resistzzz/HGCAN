import pickle
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='yoochoose1_8x', help='YC/DIG/JD/sample')
parser.add_argument('--sample_num', type=int, default=12)
parser.add_argument('--theta', type=int, default=2)
opt = parser.parse_args()

dataset = opt.dataset
sample_num = opt.sample_num

seq = pickle.load(open('datasets/' + dataset + '/all_train_seq.txt', 'rb'))
seq_category = pickle.load(open('datasets/' + dataset + '/all_train_seq_cate.txt', 'rb'))
cate2item = pickle.load(open('datasets/' + opt.dataset + '/cate2item.txt', 'rb'))


'''
def preprocess(data):
    input: raw_seq
    output：    data，Change the id to an index starting from 1
                num_items  
                max_length  
                item2idx ；{37151：1， ...}
                idx2item ：{0: 'PAD'，1:37151，....}
'''
def preprocess(data):

    PAD_Token = 0
    item_recorder = []
    idx2item = {PAD_Token: 'PAD'}
    num_items = 1
    max_length = 0

    for seq in data:
        if len(seq) > max_length:
            max_length = len(seq)
        for item in seq:
            if item not in item_recorder:
                item_recorder += [item]
                num_items += 1
    print(max(item_recorder))

    return  num_items, max_length


num, max_length = preprocess(seq)
num_cate, max_length_cate = preprocess(seq_category)
print(num)

uni_cate  = seq_category


relation = [] #Record 1-3 hops neighbors[[1, 2], [2, 1], [3, 4] , [4, 3]....]
relation_cate = []
neighbor = [] * num
cate_neighbor = [] *  num_cate

all_test = set()


adj1 = [dict() for _ in range(num)]
adj1_cate = [dict() for _ in range(num_cate)]
adj = [[] for _ in range(num)]
adj_cate = [[] for _ in range(num_cate)]
adj_c2i = [[] for _ in range(num_cate)]

for i in range(len(cate2item)):
    adj_c2i[i+1] =  cate2item[i+1]

#Iterate over sessions, find neighbors within three orders, and construct relation
for i in range(len(seq)):
    data = seq[i]
    for k in range(1, 2 * opt.theta):
        for j in range(len(data)-k):
            relation.append([data[j], data[j+k]])
            relation.append([data[j+k], data[j]])

for i in range(len(uni_cate)):
    data_cate = uni_cate[i]
    if len(data_cate) == 1:
        relation_cate.append([data_cate[0], data_cate[0]])
    for k in range(1, 2 * opt.theta):
        for j in range(len(data_cate)-k):
            relation_cate.append([data_cate[j], data_cate[j+k]])
            relation_cate.append([data_cate[j+k], data_cate[j]])


for tup in relation:
    if tup[1] in adj1[tup[0]].keys():
        adj1[tup[0]][tup[1]] += 1
    else:
        adj1[tup[0]][tup[1]] = 1

for tup in relation_cate:
    if tup[1] in adj1_cate[tup[0]].keys():
        adj1_cate[tup[0]][tup[1]] += 1
    else:
        adj1_cate[tup[0]][tup[1]] = 1



weight = [[] for _ in range(num)] #recoeda the number of occurrences of the neighbors
weight_c2c  = [[] for _ in range(num_cate)]
weight_c2i = [[] for _ in range(num_cate)]
neig_num = 0

# Take the data from adj1 and sort it in descending order of occurrence, with 'adj' storing neighbor item ids and 'weigh' storing occurrence counts
for t in range(num):
    x = [v for v in sorted(adj1[t].items(), reverse=True, key=lambda x: x[1])]
    adj[t] = [v[0] for v in x]
    weight[t] = [v[1] for v in x]
    neig_num += len(adj[t])
print('item mean neighbors: ', neig_num / num)


neig_num_cate = 0
for t in range(num_cate):
    x = [v for v in sorted(adj1_cate[t].items(), reverse=True, key=lambda x: x[1])]
    adj_cate[t] = [v[0] for v in x]
    weight_c2c[t] = [v[1] for v in x]
    neig_num_cate += len(adj_cate[t])
print('category mean neighbors: ', neig_num_cate / num_cate )

for t in range(num_cate):
    length_cate = len(adj_c2i[t])
    if length_cate != 0:
        weight_c2i[t] = [1/length_cate] * length_cate

# Sample the top 12 most frequent neighbors
for i in range(num):
    adj[i] = adj[i][:sample_num]
    weight[i] = weight[i][:sample_num]

for i in range(num_cate):
    adj_cate[i] = adj_cate[i][:sample_num]
    weight_c2c[i] = weight_c2c[i][:sample_num]

# Store an item adjacency matrix
counter_nonzero = 0
adj_item_numpy = np.identity(num , dtype= np.int)
degree_adj_item = np.zeros((num, num) , dtype = np.int)
for i in range(1, num):
    for idx, val in adj1[i].items():
        adj_item_numpy[i][idx] = 1
        counter_nonzero += 1
print(len(np.nonzero(adj_item_numpy)))
ratio = 1 - counter_nonzero/ (num * num)
print('item-item sparse rate：%0.6f %%' % (100. * ratio))

# Store an category adjacency matrix
counter_nonzero = 0
adj_cate_numpy = np.identity(num_cate , dtype= np.int)
for i in range(1, num_cate):
    for idx, val in adj1_cate[i].items():
        adj_item_numpy[i][idx] = 1
        counter_nonzero += 1
print(len(np.nonzero(adj_item_numpy)))
ratio = 1 - counter_nonzero/ (num_cate * num_cate)
print('cate-cate sparse rate：%0.6f %%' % (100. * ratio))

# Store an category2item adjacency matrix
counter_nonzero = 0
adj_c2i_numpy = np.zeros((num_cate , num) , dtype= np.int)
for i in range(1, num_cate):
    for idx in adj_c2i[i]:
        adj_c2i_numpy[i][idx] = 1
        counter_nonzero += 1
print(len(np.nonzero(adj_c2i_numpy)))
ratio = 1 - counter_nonzero/ (num_cate * num)
print('cate-item sparse rate：%0.6f %%' % (100. * ratio))

# normalize weight to [0, 1]
for i in range(1, num):
    if len(weight[i]) != 0:
        weight[i] = (np.asarray(weight[i]) / np.sum(weight[i])).tolist()


for i in range(1, num_cate):
    if len(weight_c2c[i]) != 0:
        weight_c2c[i] = (np.asarray(weight_c2c[i]) / np.sum(weight_c2c[i])).tolist()

param = {
    'num_items': num,
    'max_length': max_length,
    'num_cate': num_cate,

}

pickle.dump(adj, open('datasets/' + dataset + '/adj_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight, open('datasets/' + dataset + '/num_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(adj_cate, open('datasets/' + dataset + '/adj_cate_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight_c2c, open('datasets/' + dataset + '/num_cate_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(adj_c2i, open('datasets/' + dataset + '/adj_c2i_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(weight_c2i, open('datasets/' + dataset + '/num_c2i_' + str(sample_num) + '.pkl', 'wb'))
pickle.dump(param, open('datasets/' + dataset + '/parm.pkl', 'wb'))

