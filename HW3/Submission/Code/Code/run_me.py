import numpy as np
import kaggle
from sklearn.metrics import accuracy_score

##############################################################################################
# Read in train and test synthetic data
def load_synthetic_data():
    print('Reading synthetic data ...')
    train_x = np.loadtxt('../../Data/Synthetic/data_train.txt', delimiter = ',', dtype=float)
    train_y = np.loadtxt('../../Data/Synthetic/label_train.txt', delimiter = ',', dtype=float)
    test_x = np.loadtxt('../../Data/Synthetic/data_test.txt', delimiter = ',', dtype=float)
    test_y = np.loadtxt('../../Data/Synthetic/label_test.txt', delimiter = ',', dtype=float)
    return (train_x, train_y, test_x, test_y)
###############################################################################################

################################################################
# load corona data
def load_corona_data():
    train_x = np.load('../../Data/Corona/train_x.npy')
    train_y = np.load('../../Data/Corona/train_y.npy')
    test_x = np.load('../../Data/Corona/test_x.npy')
    return train_x, train_y, test_x
################################################################

################################################################
# load housing data
def load_housing_data():
    train_x = np.load('../../Data/Housing/train_x.npy')
    train_y = np.load('../../Data/Housing/train_y.npy')
    test_x = np.load('../../Data/Housing/test_x.npy')
    return train_x, train_y, test_x
################################################################

################################################################
# Compute MSE
def compute_MSE(y, y_hat):
        # mean squared error
        return np.mean(np.power(y - y_hat, 2))
################################################################

train_s_x, train_s_y, test_s_x, test_s_y = load_synthetic_data()

print('Train=', train_s_x.shape)
print('Test=', test_s_x.shape)
#print(train_s_x)

# get the housing data
train_h_x, train_h_y, test_h_x = load_housing_data()
print('train data shape: ', train_h_x.shape)
print('train label shape: ', train_h_y.shape)
print('test data shape: ', test_h_x.shape)

print(train_h_x)
test_h_y = np.ones(test_h_x.shape[0])
predicted_h_y = np.random.randint(0, 4, test_h_x.shape[0])

print('MSE=%0.4f' % compute_MSE(test_h_y, predicted_h_y))

# Output file location
file_name = '../Predictions/Housing/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_h_y, file_name, True)

train_c_x, train_c_y, test_c_x  = load_corona_data()
print('Train=', train_c_x.shape)
print('Test=', test_c_x.shape)

test_c_y = np.ones(test_h_x.shape[0])
predicted_c_y = np.random.randint(0, 4, test_h_x.shape[0])

print('Accuracy=%0.4f' % accuracy_score(test_c_y, predicted_c_y, normalize=True))

# Output file location
file_name = '../Predictions/Corona/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_c_y, file_name, False)
