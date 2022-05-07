import os

import numpy as np
from sklearn.datasets import load_files

IMDB_DATADIR = './aclImdb1'
classes = ["pos", "neg"]
train_data = load_files(
    os.path.join(IMDB_DATADIR, "train"), shuffle=True, categories=classes
)
# test_data = load_files(
#     os.path.join(IMDB_DATADIR, "test"), shuffle=False, categories=classes
# )

x_train = np.array(train_data.data)
y_train = np.array(train_data.target)
# x_test = np.array(test_data.data)
# y_test = np.array(test_data.target)


print(x_train.shape)  # (25000,)
print(y_train.shape)  # (25000, 1)
print(x_train[0][:50])  # this film was just brilliant casting