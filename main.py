import matplotlib.pyplot as plt
import numpy

from cross_entropy_classifier import CrossEntropyClassifier
from utils import load_data, write_output

print("Loading data")
Xtrain, Ytrain, Xtest = load_data()
Xtrain = numpy.reshape(Xtrain, (Xtrain.shape[0], -1))
Xtest = numpy.reshape(Xtest, (Xtest.shape[0], -1))

print("Fitting on training data")
model = CrossEntropyClassifier(10)
iterations = 300
history = model.fit(Xtrain, Ytrain, iterations, 0.1, 0.2)

plt.plot(range(iterations + 1), history['loss'], range(iterations + 1), history['val_loss'])
plt.legend(['train', 'validation'], loc='upper left')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

best = numpy.argmin(history['val_loss'])
model = CrossEntropyClassifier(10)
history = model.fit(Xtrain, Ytrain, best, 0.1)

print("Predicting on test data")
Ytest = model.predict(Xtest)
write_output(Ytest, 'results/Yte.csv')
