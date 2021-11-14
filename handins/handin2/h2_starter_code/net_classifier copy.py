import numpy as np

def one_in_k_encoding(vec, k):
    """ One-in-k encoding of vector to k classes 
    
    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc

def softmax(X):
    """ 
    You can take this from handin I
    Compute the softmax of each row of an input matrix (2D numpy array). 
    
    the numpy functions amax, log, exp, sum may come in handy as well as the keepdims=True option and the axis option.
    Remember to handle the numerical problems as discussed in the description.
    You should compute lg softmax first and then exponentiate 
    
    More precisely this is what you must do.
    
    For each row x do:
    compute max of x
    compute the log of the denominator sum for softmax but subtracting out the max i.e (log sum exp x-max) + max
    compute log of the softmax: x - logsum
    exponentiate that
    
    You can do all of it without for loops using numpys vectorized operations.

    Args:
        X: numpy array shape (n, d) each row is a data point
    Returns:
        res: numpy array shape (n, d)  where each row is the softmax transformation of the corresponding row in X i.e res[i, :] = softmax(X[i, :])
    """
    res = np.zeros(X.shape)
    ### YOUR CODE HERE
    max=np.amax(X,1)
    logsum=np.log(np.sum(np.exp(X-max.reshape(-1,1)),1))+max
    log_of_softmax=X-logsum.reshape(-1,1)
    res=np.exp(log_of_softmax)
    ### END CODE
    return res

def relu(x):
    """ Compute the relu activation function on every element of the input
    
        Args:
            x: np.array
        Returns:
            res: np.array same shape as x
        Beware of np.max and look at np.maximum
    """
    ### YOUR CODE HERE
    res=np.maximum(x,0)
    ### END CODE
    return res

def make_dict(W1, b1, W2, b2):
    """ Trivial helper function """
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


def get_init_params(input_dim, hidden_size, output_size):
    """ Initializer function using Xavier/he et al Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

    Args:
      input_dim: int
      hidden_size: int
      output_size: int
    Returns:
       dict of randomly initialized parameter matrices.
    """
    W1 = np.random.normal(0, np.sqrt(2./(input_dim+hidden_size)), size=(input_dim, hidden_size))
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.normal(0, np.sqrt(4./(hidden_size+output_size)), size=(hidden_size, output_size))
    b2 = np.zeros((1, output_size))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

  
class NetClassifier():
    
    def __init__(self):
        """ Trivial Init """
        self.params = None
        self.hist = None

    def predict(self, X, params=None):
        """ Compute class prediction for all data points in class X
        
        Args:
            X: np.array shape n, d
            params: dict of params to use (if none use stored params)
        Returns:
            np.array shape n, 1
        """
        if params is None:
            params = self.params
        pred = None
        ### YOUR CODE HERE
        pred=np.argmax(params['b2']+relu(params['b1']+X@params['W1'])@params['W2'],1)
        ### END CODE
        return pred
     
    def score(self, X, y, params=None):
        """ Compute accuracy of model on data X with labels y
        
        Args:
            X: np.array shape n, d
            y: np.array shape n, 1
            params: dict of params to use (if none use stored params)

        Returns:
            np.array shape n, 1
        """
        if params is None:
            params = self.params
        acc = None
        ### YOUR CODE HERE
        acc=np.average(self.predict(X,params)==y)
        ### END CODE
        return acc
    
    @staticmethod
    def cost_grad(X, y, params, c=0.0):
        """ Compute cost and gradient of neural net on data X with labels y using weight decay parameter c
        You should implement a forward pass and store the intermediate results 
        and the implement the backwards pass using the intermediate stored results
        
        Use the derivative for cost as a function for input to softmax as derived above
        
        Args:
            X: np.array shape n, self.input_size
            y: np.array shape n, 1
            params: dict with keys (W1, W2, b1, b2)
            c: float - weight decay parameter
            params: dict of params to use for the computation
        
        Returns 
            cost: scalar - average cross entropy cost
            dict with keys
            d_w1: np.array shape w1.shape, entry d_w1[i, j] = \partial cost/ \partial W1[i, j]
            d_w2: np.array shape w2.shape, entry d_w2[i, j] = \partial cost/ \partial W2[i, j]
            d_b1: np.array shape b1.shape, entry d_b1[1, j] = \partial cost/ \partial b1[1, j]
            d_b2: np.array shape b2.shape, entry d_b2[1, j] = \partial cost/ \partial b2[1, j]
            
        """
        
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']
        k=W2.shape[1]
        labels = one_in_k_encoding(y,k) # shape n x k
                        
        ### YOUR CODE HERE - FORWARD PASS - compute cost with weight decay and store relevant values for backprop
        n=X.shape[0]
        m=X.shape[1]
        h=W1.shape[1]
        cost=c

        a=X@W1 #1xm, mxh, 1xh
        b=b1+a #1xh
        c=relu(b) #1xh
        d=c@W2 #1xh, hxk, 1xk
        z=b2+d #1xk
        
        softmax_z=softmax(z)
        vector=np.zeros(y.shape)
        for i in range(y.shape[0]):
            vector[i]=labels[i,:]@softmax_z[i,:]
        LV=np.log(vector)
        #L vectorized and not averaged
        L=-np.average(LV)+cost*(np.sum(W1**2)+np.sum(W2**2))
        ### END CODE
        
        ### YOUR CODE HERE - BACKWARDS PASS - compute derivatives of all weights and bias, store them in d_w1, d_w2' d_w2, d_b1, d_b2
        d_w1=[]
        d_w2=[]
        d_b1=[]
        d_b2=[]
        for iter in np.arange(n):
            d_L=1
            d_z=d_L*(softmax_z[iter,:] - labels[iter,:]) #1xk
            d_b2.append(d_z) #1xk
            d_d=d_z #1xk
            store_w2=np.zeros([h,k])
            for i in np.arange(h):
                store_w2[i,:]=c[iter,i]*d_d
            d_w2.append(store_w2)
            d_c=d_d@np.transpose(W2)#1xh
            d_b=d_c@(np.diag(b[iter,:].reshape(-1)>0))#1xh
            d_b1.append(d_b)
            d_a=d_b #1xh
            store_w1=np.zeros([m,h])
            for i in np.arange(m):
                store_w1[i,:]=X[iter,i]*d_a
            d_w1.append(store_w1)
        d_w1=sum(d_w1)/n+2*cost*W1
        d_w2=sum(d_w2)/n+2*cost*W2
        d_b1=(sum(d_b1)/n).reshape(1,-1)
        d_b2=(sum(d_b2)/n).reshape(1,-1)
        ### END CODE
        # the return signature
        return L, {'d_w1': d_w1, 'd_w2': d_w2, 'd_b1': d_b1, 'd_b2': d_b2}
        
    def fit(self, X_train, y_train, X_val, y_val, init_params, batch_size=32, lr=0.1, c=1e-4, epochs=30):
        """ Run Mini-Batch Gradient Descent on data X, Y to minimize the in sample error for Neural Net classification
        Printing the performance every epoch is a good idea to see if the algorithm is working
    
        Args:
           X_train: numpy array shape (n, d) - the training data each row is a data point
           y_train: numpy array shape (n,) int - training target labels numbers in {0, 1,..., k-1}
           X_val: numpy array shape (n, d) - the validation data each row is a data point
           y_val: numpy array shape (n,) int - validation target labels numbers in {0, 1,..., k-1}
           init_params: dict - has initial setting of parameters
           lr: scalar - initial learning rate
           batch_size: scalar - size of mini-batch
           c: scalar - weight decay parameter 
           epochs: scalar - number of iterations through the data to use

        Sets: 
           params: dict with keys {W1, W2, b1, b2} parameters for neural net
        returns
           hist: dict:{keys: train_loss, train_acc, val_loss, val_acc} each an np.array of size epochs of the the given cost after every epoch
           loss is the NLL loss and acc is accuracy
        """
        
        W1 = init_params['W1']
        b1 = init_params['b1']
        W2 = init_params['W2']
        b2 = init_params['b2']
        hist = {
            'train_loss': None,
            'train_acc': None,
            'val_loss': None,
            'val_acc': None, 
        }

        
        ### YOUR CODE HERE
        
        ### END CODE
        # hist dict should look like this with something different than none
        #hist = {'train_loss': None, 'train_acc': None, 'val_loss': None, 'val_acc': None}
        ## self.params should look like this with something better than none, i.e. the best parameters found.
        # self.params = {'W1': None, 'b1': None, 'W2': None, 'b2': None}
        return hist
        

def numerical_grad_check(f, x, key):
    """ Numerical Gradient Checker """
    eps = 1e-6
    h = 1e-5
    # d = x.shape[0]
    cost, grad = f(x)
    grad = grad[key]
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:    
        dim = it.multi_index    
        tmp = x[dim]
        x[dim] = tmp + h
        cplus, _ = f(x)
        x[dim] = tmp - h 
        cminus, _ = f(x)
        x[dim] = tmp
        num_grad = (cplus-cminus)/(2*h)
        # print('cplus cminus', cplus, cminus, cplus-cminus)
        # print('dim, grad, num_grad, grad-num_grad', dim, grad[dim], num_grad, grad[dim]-num_grad)
        assert np.abs(num_grad - grad[dim]) < eps, 'numerical gradient error index {0}, numerical gradient {1}, computed gradient {2}'.format(dim, num_grad, grad[dim])
        it.iternext()

def test_grad():
    stars = '*'*5
    print(stars, 'Testing  Cost and Gradient Together')
    input_dim = 7
    hidden_size = 1
    output_size = 3
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)

    nc = NetClassifier()
    X = np.random.randn(7, input_dim)
    y = np.array([0, 1, 2, 0, 1, 2, 0])

    f = lambda z: nc.cost_grad(X, y, params, c=1.0)
    print('\n', stars, 'Test Cost and Gradient of b2', stars)
    numerical_grad_check(f, params['b2'], 'd_b2')
    print(stars, 'Test Success', stars)
    
    print('\n', stars, 'Test Cost and Gradient of w2', stars)
    numerical_grad_check(f, params['W2'], 'd_w2')
    print('Test Success')
    
    print('\n', stars, 'Test Cost and Gradient of b1', stars)
    numerical_grad_check(f, params['b1'], 'd_b1')
    print('Test Success')
    
    print('\n', stars, 'Test Cost and Gradient of w1', stars)
    numerical_grad_check(f, params['W1'], 'd_w1')
    print('Test Success')

if __name__ == '__main__':
    input_dim = 3
    hidden_size = 5
    output_size = 4
    batch_size = 7
    nc = NetClassifier()
    params = get_init_params(input_dim, hidden_size, output_size)
    X = np.random.randn(batch_size, input_dim)
    Y = np.array([0, 1, 2, 0, 1, 2, 0])
    nc.cost_grad(X, Y, params, c=0)
    test_grad()
