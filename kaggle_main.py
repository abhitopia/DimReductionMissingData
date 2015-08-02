from utils import convert_kaggle_data_to_npy, load_data_set
from DimReductionMIssingData import DimReductionMissingData


def convert_to_numpy():
    convert_kaggle_data_to_npy('train')
    convert_kaggle_data_to_npy('test')


def main():
    #  convert_to_numpy()
    print 'Loading data...'
    data = load_data_set('Data/trainData.npy')

    obj = DimReductionMissingData(data, reduced_dim=100)
    obj.optimize(num_epochs=5, batch_size=1000)

if __name__ == '__main__':
    main()




