# Import python modules
import numpy as np
import kaggle

# Read in train and test data
def read_dataA():
	print('Reading air foil dataset ...')
	train_data = np.load('../../Data/dataA/train.npy')
	train_x = train_data[:,0:train_data.shape[1]-1]
	train_y = train_data[:,train_data.shape[1]-1]
	test_data = np.load('../../Data/dataA/test.npy')
	test_x = test_data

	return (train_x, train_y, test_x)

def read_dataB():
    print('Reading air quality dataset ...')
    train_data = np.load('../../Data/dataB/train.npy')
    train_x = train_data[:,0:train_data.shape[1]-1]
    train_y = train_data[:,train_data.shape[1]-1]
    test_data = np.load('../../Data/dataB/test.npy')
    test_x = test_data
	
    return (train_x, train_y, test_x)

# Compute MAE
def compute_error(y_hat, y):
	# mean absolute error
	return np.abs(y_hat - y).mean()

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

############################################################################

train_x, train_y, test_x = read_dataA()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

train_x, train_y, test_x = read_dataB()
print('Train=', train_x.shape)
print('Test=', test_x.shape)

# Create dummy test output values
predicted_y = np.ones(test_x.shape[0]) * -1
# Output file location
file_name = '../Predictions/dataA/best.csv'
# Writing output in Kaggle format
print('Writing output to ', file_name)
kaggle.kaggleize(predicted_y, file_name)
