import matplotlib
matplotlib.use('Agg') # disable pop up windows

import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_data

print("Loading data")
Xtrain, Ytrain, Xtest = load_data()

plot_train = False
plot_test = False

if plot_train:
    print("Plotting training images")
    for i in tqdm(range(len(Xtrain))):
        plt.imshow(Xtrain[i,:,:,:] * 2.5 + 0.5, interpolation='none')
        plt.savefig('plots/tr' + str(i) + '_' + str(Ytrain[i]) + '.png')
     
if plot_test:
    print("Plotting testing images")
    for i in tqdm(range(len(Xtest))):
        plt.imshow(Xtest[i,:,:,:] * 2.5 + 0.5, interpolation='none')
        plt.savefig('plots/te' + str(i) + '.png')
