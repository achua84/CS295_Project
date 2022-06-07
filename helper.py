from collections import defaultdict

num_channels = [] 
feature_sizes = []
P_matrices = []
m = 10000 # Number of dimensions (can change)
ID_length = 2000 # Number of ID images used for testing (can change)
OOD_length = 1000 # Number of OOD images used for testing, x2 since there is near_OOD and far_OOD (can change)
num_layers = 60 # Number of feature maps (max is 71, can change)


weights = []
class_descriptors = defaultdict(list)
ID_encodings = defaultdict(list)
near_OOD_encodings = defaultdict(list)
far_OOD_encodings = defaultdict(list)
