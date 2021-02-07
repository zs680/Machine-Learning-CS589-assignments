# Import python modules
import numpy as np
import kaggle
from sklearn.metrics import accuracy_score

# Read in train and test data
def read_image_data():
	print('Reading image data ...')
	train_x = np.load('../../Data/data_train.npy')
	train_y = np.load('../../Data/train_labels.npy')
	test_x = np.load('../../Data/data_test.npy')

	return (train_x, train_y, test_x)

############################################################################


def read_image_data_knn():
	print('Reading image data ...')
	train_x = np.load('../../Data/data_train_knn.npy')
	train_y = np.load('../../Data/train_labels_knn.npy')
	test_x = np.load('../../Data/data_test.npy')

	return (train_x, train_y, test_x)

############################################################################
index=[]
set=[]
def  split_train_test(data, K):
    shuffled_indices = np.random.permutation(data)
    #print(shuffled_indices)
    fold_size = int(data//K)
    #print(fold_size)
    reminder=data%K
    #print(reminder)
    #print(range(0,reminder))
    for i in range(reminder):

        fold= shuffled_indices[i*fold_size+i:(i+1)*fold_size+i+1]
        #print(i,fold)
        set.append(fold)

    for i in range(reminder,K):
        fold = shuffled_indices[i * fold_size + reminder:(i + 1) * fold_size + reminder ]
        #print(i,fold)
        set.append(fold)


    for j in range(len(set)):
        train_indicis = []
        test_indicies = []
        for a in range(len(set)):
            if a != j:
                for k in set[a]:
                    train_indicis.append(k)
            if a == j:
                for k in set[a]:
                    test_indicies.append(k)
        index.append((train_indicis, test_indicies))
    return index
#####################################################################################
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()
def Average(lst):
    return sum(lst) / len(lst)

##################################################
train_x, train_y, test_x = read_image_data()

train_knn_x, train_knn_y, test_knn_x = read_image_data_knn()


# Create dummy test output values to compute accuracy
test_y = np.ones(test_x.shape[0])
predicted_y = np.random.randint(0, 4, test_x.shape[0])
print('DUMMY Accuracy=%0.4f' % accuracy_score(test_y, predicted_y, normalize=True))

#test_y = np.load('../../../test_labels_hw2.npy')
predicted_y = np.copy(test_y)
# Output file location
file_name = '../Predictions/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)

