import math
import numpy as np

import os
filename = ''

if not os.path.exists(filename):
    print(f'Oops! We failed to find the data file {filename}')
else:
    data = np.load(filename, allow_pickle=True).item()
    
    for k, v in data.items():
        print(f'Key "{k}" has shape {v.shape}')
        print(f'First three rows of {k} are {v[:3]}')

from soft_margin_svm import svm_train_bgd, svm_test

W, b = svm_train_bgd(data['x_train'], data['y_train'])

if np.all(W == 0):
    print('you should update W')
if np.all(b == 0):
    print('you should update b')
    
accuracy = svm_test(W, b, data['test'], data['test'])
print(f'The accuracy of batch gradient descent-based SVM is {accuracy*100:4.2f}%')


from soft_margin_svm import svm_train_bgd, svm_test

for num_epochs in [1, 3, 10, 30, 100]:
    W, b = svm_train_bgd(data['x_train'], data['y_train'], num_epochs)
    accuracy = svm_test(W, b, data['test'], data['y_test'])
    print(f'[NumEpochs: {num_epochs}] Accuracy: {accuracy * 100:4.2f}%')
    print(f'b: {b}, W: {W}')