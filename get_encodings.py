import pickle 
import os
import random


class_descriptors = None
ID_encodings = None
near_OOD_encodings = None
far_OOD_encodings = None


if os.path.exists('all_encodings.pkl'):
    with open('all_encodings.pkl', 'rb') as f:
        class_descriptors, ID_encodings, \
        near_OOD_encodings, far_OOD_encodings = pickle.load(f)

near_ID_merge = []
far_ID_merge = []

near_list = []
far_list = []
# Combine all encodings in the form (encoding, target_value)
# in which
# 0: in distribution
# 1: out of distribution 

for key, vals in ID_encodings.items():
    for val in vals: 
        near_ID_merge.append((val, 0))
        far_ID_merge.append((val, 0))

for key, vals in near_OOD_encodings.items():
    for val in vals: 
        near_ID_merge.append((val, 1))
        near_list.append((val, 1))

for key, vals in far_OOD_encodings.items():
    for val in vals: 
        far_ID_merge.append((val, 1))
        far_list.append((val, 1))

random.shuffle(near_ID_merge)
random.shuffle(far_ID_merge)
print(len(near_ID_merge))
print(len(far_ID_merge))

for i in range(2):
    print(type(near_ID_merge[i]), " ", len(near_ID_merge[i][0]), " ", near_ID_merge[i][:20])

for i in range(2):
    print(type(far_ID_merge[i]), " ", len(far_ID_merge[i][0]), " ", far_ID_merge[i][:20])