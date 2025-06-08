import numpy as np
import os

ab_path = os.getcwd() + '/datasets/'
data_directory=ab_path + 'CCTA_GT/'

np.random.seed(1)
#######################################
# Declare Parameters
SAMPLES_PARA = {}

SAMPLES_PARA['num_phantoms'] = len(os.listdir(ab_path + 'CCTA_GT/'))
SAMPLES_PARA['gen_filter_num']=32
# 75% training data, 15% validation data and 10% test data
random_index = np.random.choice((np.arange(SAMPLES_PARA['num_phantoms'])+1),int(SAMPLES_PARA['num_phantoms']*25/100),False).tolist()

SAMPLES_PARA['validation_index'] = random_index[:int(SAMPLES_PARA['num_phantoms']*15/100)]
SAMPLES_PARA['test_index'] = random_index[:(int(SAMPLES_PARA['num_phantoms']*15/100)-1):-1]
SAMPLES_PARA['train_index'] = [i for i in ((np.arange(SAMPLES_PARA['num_phantoms'])+1).tolist())
                    if (i not in SAMPLES_PARA['validation_index']) & (i not in SAMPLES_PARA['test_index'])]

# check if the files exists
SAMPLES_PARA['validation_index'] = [i for i in SAMPLES_PARA['validation_index'] if os.path.exists(os.path.join(data_directory, f"{i}.npy"))]
SAMPLES_PARA['test_index'] = [i for i in SAMPLES_PARA['test_index'] if os.path.exists(os.path.join(data_directory, f"{i}.npy"))]
SAMPLES_PARA['train_index'] = [i for i in SAMPLES_PARA['train_index'] if os.path.exists(os.path.join(data_directory, f"{i}.npy"))]

SAMPLES_PARA['num_train_data'] = len(SAMPLES_PARA['train_index'])
SAMPLES_PARA['num_validation_data'] = len(SAMPLES_PARA['validation_index'])
SAMPLES_PARA['num_test_data'] = len(SAMPLES_PARA['test_index'])