import matplotlib.pyplot as plt
import numpy

from cross_entropy_classifier import CrossEntropyClassifier
from utils import load_data, plot_history, write_output

output_suffix = 'trial1'

print("Loading data")
Xtrain, Ytrain, Xtest = load_data()
Xtrain = numpy.reshape(Xtrain, (Xtrain.shape[0], -1))
Xtest = numpy.reshape(Xtest, (Xtest.shape[0], -1))

print("Fitting on training data")
model = CrossEntropyClassifier(10)
iterations = 40
history = model.fit(Xtrain, Ytrain, iterations, 0.1, 0.2, 10)
print len(history['loss'])

best = numpy.argmax(history['val_accuracy'])
print("best accuracy is %.3f at iteration %d" % (history['val_accuracy'][best], best))

f = plot_history(history)
f.savefig('plots/' + output_suffix + '.png')

model = CrossEntropyClassifier(10)
history = model.fit(Xtrain, Ytrain, best, 0.1)

print("Predicting on test data")
Ytest = model.predict(Xtest)
write_output(Ytest, 'results/Yte_' + output_suffix + '.csv')
