# Importing the libraries
import numpy as np
import pandas as pd
import pickle
import csv
from sklearn.linear_model import LinearRegression
import os

here = os.path.dirname(os.path.abspath(__file__))

def read_input(dir):
    x, y = np.empty(0), np.empty(0)
    with open(dir, newline = '\n') as csvfile:
        csvreader = csv.reader(csvfile, delimiter = ',')
        for row in csvreader:
            x = np.append(x, float(row[0]))
            y = np.append(y, float(row[1]))
    return x, y

def main():
    model = LinearRegression()
    x, y = read_input("./dataset/A")
    #Fitting model with trainig data
    model.fit(x.reshape(-1,1), y)

    # Saving model to disk
    with open(os.path.join(here, 'model.pkl'),'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
	main()