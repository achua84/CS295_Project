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

all_encodings = []

# Combine all encodings in the form (encoding, target_value)
# in which
# 0: in distribution
# 1: out of distribution 
for key, vals in ID_encodings.items():
	for val in vals: 
		all_encodings.append((val, 0))

for key, vals in near_OOD_encodings.items():
	for val in vals: 
		all_encodings.append((val, 1))

for key, vals in far_OOD_encodings.items():
	for val in vals: 
		all_encodings.append((val, 1))

random.shuffle(all_encodings)
print(len(all_encodings))

for i in range(5):
	print(type(all_encodings[i]), " ", len(all_encodings[i][0]), " ", all_encodings[i][:20])