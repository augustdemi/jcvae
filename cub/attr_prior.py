import numpy as np
import pickle

path = '../../data/cub/CUB_200_2011/CUB_200_2011/'

vec_attr_trainval = pickle.load(open(path + "attributes/vec_attr_trainval.pkl", "rb"))
vec_attr_test = pickle.load(open(path + "attributes/vec_attr_test.pkl", "rb"))

total_prior = []
for i in range(28):
    prior = np.zeros_like(vec_attr_test[list(vec_attr_test.keys())[0]][i])
    for key in vec_attr_trainval.keys():
        prior += vec_attr_trainval[key][i]
    for key in vec_attr_test.keys():
        prior += vec_attr_test[key][i]
    prior = prior / prior.sum()
    total_prior.append(prior)
print(total_prior)
f = open(path + "attributes/attr_prior.pkl", "wb")
pickle.dump(total_prior, f)
f.close()

# visually distinct attributes
primary_attr = ['eye_color', 'bill_length', 'size', 'shape', 'breast_pattern', 'belly_pattern', 'bill_shape',
                'bill_color', 'throat_color', 'crown_color', 'forehead_color', 'underparts_color', 'primary_color',
                'breast_color', 'wing_color']
primary_attr_idx = []
attributes = np.genfromtxt(path + 'attributes/attr.txt', delimiter='\n', dtype=str)
for i in range(attributes.shape[0]):
    if attributes[i].split("::")[0] in primary_attr:
        primary_attr_idx.append(i)
