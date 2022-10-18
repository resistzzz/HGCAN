import pickle
import numpy as np

item2cate = pickle.load(open('item2cate.txt','rb'))
# item2cate = np.array(item2cate)
cate_count = []
for i in list(item2cate):
    if item2cate[i] not in cate_count:
        cate_count += [item2cate[i]]
cate2item = {}
for i in item2cate:
    # item2cate[i] = np.array(item2cate[i])
    if item2cate[i] not in cate2item:
        cate2item[item2cate[i]] = [i]
    else:
        cate2item[item2cate[i]].append(i)
pickle.dump(cate2item, open('cate2item.txt', 'wb'))
print('')