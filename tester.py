
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

evaluation_results = torch.load('evaluation_results')

# network_name = pd.DataFrame(np.random.randn(6, 4), index= amount_of_data, columns=dice_error))

x_test = np.load('x_test.npy')
y_pred = np.load('y_pred.npy')


for s in range(0, 10):
    plt.imshow(x_test[s,:,:,0], plt.cm.gray)
    plt.show()
    plt.imshow(y_pred[s,:,:,0], plt.cm.gray)
    plt.show()



something = 0