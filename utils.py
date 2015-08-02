import csv
import gzip
import cPickle
import numpy as np

def convert_kaggle_data_to_npy(choice='train'):
    print 'Converting data, please wait...'
    if choice is 'train':
        X = np.zeros((105471, 769), dtype=np.float32)
        y = np.zeros((105471, 1), dtype=np.float32)
        one_percent_const = 1055
        file_name = './Data/train_v2.csv'
    else:
        file_name = './Data/test_v2.csv'
        X = np.zeros((316415, 769), dtype=np.float32)
        one_percent_const = 3164
    index = 0
    with open(file_name, 'rb') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for row in reader:
            if row[0] == 'id':
                continue
            else:
                row = np.array(row)
                row[row == 'NA'] = np.nan
                if choice is 'train':
                    X[index, :] = row[1:-1]
                    y[index] = row[-1]
                else:
                    X[index, :] = row[1:]
                index += 1
                if index % one_percent_const == 0:
                    print int(index/one_percent_const), '% done'
    print 'Data Loaded!!'
    if choice is 'train':
        f = open('Data/trainData.npy', 'wb')
        np.save(f, X)
        np.save(f, y)
        f.close()
    else:
        f = open('Data/testData.npy', 'wb')
        np.save(f, X)
        f.close()


def load_data_set(file_name):
    with open(file_name, 'rb') as f:
        data = np.load(f)
    return data

def mnist_loader():
    # Load the dataset
    with gzip.open('Data/mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
    return train_set, valid_set, test_set

