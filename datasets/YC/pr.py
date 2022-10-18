import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
dataset = '../yoochoose1_8x'
# dataset = '../jdata_x'
# dataset = '../diginetica_x'



def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)

# item2cate = pickle.load(open(dataset +'/item2cate.txt','rb'))
# # item2cate = np.array(item2cate)
# # all_train_seq_cate = pickle.load(open(dataset +'/all_train_seq_cate.txt','rb'))
# train_cat = pickle.load(open(dataset +'/train_cat.txt','rb'))
# test_cat = pickle.load(open(dataset +'/test_cat.txt','rb'))
all_train_seq_cate = pickle.load(open(dataset +'/all_train_seq_cate.txt','rb'))
all_train_seq = pickle.load(open(dataset +'/all_train_seq.txt','rb'))

# train_set , valid_set = split_validation(train_cat, valid_portion = 0.1)

# ================cate transition======================
def transition(data):
    trans_record = []
    trans_num_record = []
    for seq in data:
        for i in range(len(seq)-1):
            if [seq[i],seq[i+1]] not in trans_record:
                trans_record.append([seq[i],seq[i+1]])
                index = trans_record.index([seq[i], seq[i + 1]])
                trans_num_record.append(1)
            else:
                index = trans_record.index([seq[i],seq[i+1]])
                trans_num_record[index] += 1
    ss = sorted(trans_num_record)
    return ss
a = transition(all_train_seq_cate)
b = transition(all_train_seq)
# def piece_array(a):
#     piece_a  = {}
#     for i in a:
#         log_i = int(math.log(i,2))
#         if log_i not in piece_a:
#             piece_a[log_i] = 1
#         else:
#             piece_a[log_i] += 1
#     return list(piece_a.values())
# a_s = piece_array(a)
# b_s = piece_array(b)
# plt.scatter(np.arange(len(a_s)), a_s,c = 'b')
# plt.scatter(np.arange(len(b_s)), b_s,c = 'r')
#
#
# plt.title('trans')
# plt.xlabel('transs')
# plt.ylabel('num')
#
# plt.show()
# print(i)



#============================ train set ================================================
# cate_most_appear_count = 0
# cate_appear_count = 0
# for i in range(len(train_set[0])):
#     train_seq = train_set[0][i]
#     most_appear_cate = max(set(train_seq),key = train_seq.count)
#     if most_appear_cate == train_set[1][i]:
#         cate_most_appear_count+=1
#     if train_set[1][i] in  train_set[0][i]:
#         cate_appear_count += 1
#
# print('train_set the cate most-appear probability', cate_most_appear_count/len(train_set[0]))
# print('train_set the cate appear probability', cate_appear_count/len(train_set[0]))
#
# #==============================  valid set   ====================================================
# cate_most_appear_count = 0
# cate_appear_count = 0
# for i in range(len(valid_set[0])):
#     train_seq = valid_set[0][i]
#     most_appear_cate = max(set(train_seq),key = train_seq.count)
#     if most_appear_cate == valid_set[1][i]:
#         cate_most_appear_count+=1
#     if valid_set[1][i] in  valid_set[0][i]:
#         cate_appear_count += 1
#
# print('valid_set the cate most-appear probability', cate_most_appear_count/len(valid_set[0]))
# print('valid_set the cate appear probability', cate_appear_count/len(valid_set[0]))
#
# #==============================  test set   ====================================================
# cate_most_appear_count = 0
# cate_appear_count = 0
# for i in range(len(test_cat[0])):
#     train_seq = test_cat[0][i]
#     most_appear_cate = max(set(train_seq),key = train_seq.count)
#     if most_appear_cate == test_cat[1][i]:
#         cate_most_appear_count+=1
#     if test_cat[1][i] in  test_cat[0][i]:
#         cate_appear_count += 1
#
# print('test_set the cate most-appear probability', cate_most_appear_count/len(test_cat[0]))
# print('test_set the cate appear probability', cate_appear_count/len(test_cat[0]))
