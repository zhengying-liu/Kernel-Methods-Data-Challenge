import numpy
import os
import pandas as pd

def one_hot(i, n_classes=10):
    return numpy.eye(n_classes)[i]

def ensemble_from_csv(directory="ensemble/", max_nb_models=3, save_name="ensemble.csv"):
    ensemble = []
    print("Voting from following models:")
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            print(filename)
            df = pd.read_csv(directory + filename)
            if 'Id' in df.columns and 'Prediction' in df.columns:
                ensemble.append(df)
                if len(ensemble) >= max_nb_models:
                    break
    res = ensemble[0].copy()
    for index, row in res.iterrows():
        sum_vec = numpy.zeros(10)
        for df in ensemble:
            sum_vec += one_hot(df.loc[index].Prediction)
        row.Prediction = sum_vec.argmax()
    res = res[['Id', 'Prediction']]
    res.to_csv("submission/" + save_name, index=None)
    return res
