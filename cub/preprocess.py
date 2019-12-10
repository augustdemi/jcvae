import numpy as np
import pickle

path = '../../data/cub/CUB_200_2011/CUB_200_2011/'

# find sample id for trainval / test
train_val_label = np.genfromtxt(path + "attributes/trainvalids.txt", delimiter='\n', dtype=int)
imgid_label = [int(elt.split(' ')[1]) for elt in
               np.genfromtxt(path + "attributes/image_class_labels.txt", delimiter='\n', dtype=str)]
trainval_imgid = [i + 1 for i in range(len(imgid_label)) if imgid_label[i] in train_val_label]
test_imgid = [i + 1 for i in range(len(imgid_label)) if imgid_label[i] not in train_val_label]

# make 312 attirubtes into 28 attirubtes
attributes = np.genfromtxt(path + 'attributes/attr.txt', delimiter='\n', dtype=str)
attr_len = []
for attr in attributes:
    attr_len.append(len(attr.split("::")[1].split(',')))
attr_len = np.array(attr_len)

primary_attr = ['eye_color', 'bill_length', 'size', 'shape', 'breast_pattern', 'belly_pattern', 'bill_shape',
                'bill_color', 'throat_color', 'crown_color', 'forehead_color', 'underparts_color', 'primary_color',
                'breast_color', 'wing_color']
primary_attr_idx = []

for i in range(attributes.shape[0]):
    if attributes[i].split("::")[0] in primary_attr:
        primary_attr_idx.append(i)

present_per_img = {}
ll = np.genfromtxt(path + 'attributes/image_attribute_labels.txt', delimiter='\n', dtype=str)
cnt = 0
img_idx = 1
pres = []
for l in ll:
    cnt += 1
    pres.append(int(l.split(' ')[2]))
    if cnt == 312:
        present_per_img.update({img_idx: pres})
        cnt = 0
        img_idx += 1
        pres = []

summarized_attr = {}
for k in present_per_img.keys():  # per img
    summarized_attr_one_img = {}
    for i in range(attr_len.shape[0]):  # per attribute
        aa = present_per_img[k][attr_len[:i].sum():attr_len[:i + 1].sum()]
        pres_idx = []
        for j in range(len(aa)):  # find the presented index
            if aa[j] == 1:
                pres_idx.append(j)
            summarized_attr_one_img.update({i: pres_idx})
    summarized_attr.update({k: summarized_attr_one_img})

f = open(path + "attributes/summarized_present_per_img.pkl", "wb")
pickle.dump(summarized_attr, f)
f.close()

# per img sample, make vec_attribute for each of 28 attributes.(trainval/ test split)
summarized_attr = pickle.load(open(path + "attributes/summarized_present_per_img.pkl", "rb"))
vec_attr_trainval = {}
vec_attr_test = {}
for sampleid in summarized_attr.keys():
    attr_per_sample = summarized_attr[sampleid]
    vec_attr_per_sample = []
    for i in range(28):
        one_vec_attr = np.zeros(attr_len[i] + 1)  # in case an attribute was invisible, add one more dim
        for idx in attr_per_sample[i]:
            one_vec_attr[idx] = 1
        if one_vec_attr.sum() == 0:  # in case an attribute was invisible
            one_vec_attr[-1] = 1
        vec_attr_per_sample.append(one_vec_attr)
    if sampleid in trainval_imgid:
        vec_attr_trainval.update({sampleid: vec_attr_per_sample})
    else:
        vec_attr_test.update({sampleid: vec_attr_per_sample})

f = open(path + "attributes/vec_attr_trainval.pkl", "wb")
pickle.dump(vec_attr_trainval, f)
f.close()
f = open(path + "attributes/vec_attr_test.pkl", "wb")
pickle.dump(vec_attr_test, f)
f.close()
