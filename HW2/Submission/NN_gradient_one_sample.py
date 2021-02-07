import autograd.numpy as np
import autograd

# sigmoid function
sig  = np.tanh
sigp = lambda x : 1 - np.tanh(x)**2

# Loss function
def L(x, y, W, V, b, c):
	l = c + V @ sig(b + W @ x)
	return -l[y] + np.log(np.sum(np.exp(l)))

# Partial derivatives
def partial_derivatives(x, y, W, V, b, c):
	# Filling in some dummy values
	# THIS IS WHERE YOU WILL WRITE YOUR PARTIAL DERIVATIVES
	e=np.zeros(c.shape)
	def e_hat(y):
		e=np.zeros((dims_out, 1))
		for i in range(0,dims_out):
			if i==y:
				e[i]=1
		return e
	def g(T):
		for i in range(0,c.shape[0]):
			e[i]=np.exp(T[i])/(np.sum(np.exp(T)))
		return e

	l = c + V @ sig(b + W @ x)
	h=sig(b + W @ x)
	s=b + W @ x
	dLdl=e_hat(y)+g(l)
	dLdc=dLdl
	dLdV = dLdl@h.T
	dLdh=V.T@dLdl
	dLds=dLdh * sigp(s)
	dLdb = dLds
	dLdW=sigp(s)* (V.T@dLdl)@x.T
	return dLdW, dLdV, dLdb, dLdc





# DO NOT REMOVE OR UNCOMMENT THIS LINE OF CODE
# setting random seed for reproducibility
seed = 356
np.random.seed(seed)

# Loading the input
x = np.load('nn_gradient_sample.npy')
# Number of input dimensions
dims_in  = x.shape[0]
# Setting label
y = np.array([2])
# Number of output dimensions
dims_out = 4

# Number of hidden units
M=5
dims_hid = M

# Initializing weights
W = np.random.randn(dims_hid, dims_in)
b = np.random.randn(dims_hid, 1)
V = np.random.randn(dims_out, dims_hid)
c = np.random.randn(dims_out, 1)

# Computing partial derivatives
#calculation of partial derivatives:
dLdW_pd, dLdV_pd, dLdb_pd, dLdc_pd = partial_derivatives(x, y, W, V, b, c)







np.set_printoptions(precision=6)

# print loss
print('Loss = %0.4f' % L(x, y, W, V, b, c))

# print partial derivatives
# Computing partial derivatives using autograd. L is the loss function and 5 is the position of the c
dLdc_autograd = autograd.grad(L, 5)
print('dLdc, Autograd\n', dLdc_autograd(x, y, W, V, b, c).T)
print('dLdc, partial derivative\n', dLdc_pd.T)

# Computing partial derivatives using autograd. L is the loss function and 3 is the position of the V
dLdV_autograd = autograd.grad(L, 3)
print('dLdV, Autograd\n', dLdV_autograd(x, y, W, V, b, c))
print('dLdV, partial derivative\n', dLdV_pd)

# Computing partial derivatives using autograd. L is the loss function and 4 is the position of the b
dLdb_autograd = autograd.grad(L, 4)
print('dLdb, Autograd\n', dLdb_autograd(x, y, W, V, b, c).T)
print('dLdb, partial derivative\n', dLdb_pd.T)

# Computing partial derivatives using autograd. L is the loss function and 2 is the position of the W
dLdW_autograd = autograd.grad(L, 2)
# Due to space limitations we are only printing few values of W
to_print_rows = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
to_print_cols = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
print('dLdW, Autograd\n', dLdW_autograd(x, y, W, V, b, c)[to_print_rows, to_print_cols])
print('dLdW, partial derivative\n', dLdW_pd[to_print_rows, to_print_cols])

