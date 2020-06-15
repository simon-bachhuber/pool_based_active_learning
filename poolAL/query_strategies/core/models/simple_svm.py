import numpy as np
from .model import Model
from ..utils import sort_by_2nd, replace_labels

def sigma(x):
    global alpha
    return 1/(1+np.exp(-alpha*x))

class rbf_kernel:
    def __init__(self, gamma):
        self.gamma = gamma

    def calc(self, x1, x2):
        return np.exp(-self.gamma*np.sum((x1-x2)**2))

class SVM(Model):
    '''
    SVM with linear and rbf kernel

    Parameters:
    -----------

    C: Parameter for mu adaption. See References for more info.
        Rough estimate C=12 * rate_of_wrong_labels. Suppose on average 40% of labels are incorrect, then C = 12*0.4 
        default = 7

    alpha: {float}, Parameter for sigma function
        default = 10

    gamma: {float}, Parameter for rbf kernel
        default = 2

    kernel: {string}, either 'linear' or 'rbf'
        default = 'linear'

    learning_rate: {float}
        default = 0.2

    conv_crit: {float}
        default = 10**-10

    n_iterations: {int}
        default = 500

    Methods:
    --------

    .train(Dataset)

    .predict(X)

    .score(Dataset)

    References:
    ----------
        [1]   Xuejun Liao, Ya Xue, Lawrence Carin. Logistic Regression with an Auxiliary Data Source.

    '''

    def __init__(self, **kwargs):

        self.alpha = kwargs.pop('alpha', 1)
        global alpha
        alpha = self.alpha

        # Which kernel to use, available 'linear', 'rbf', 'poly'
        self.kernel = kwargs.pop('kernel', 'linear')


        if self.kernel is 'rbf':
            self.gamma = kwargs.pop('gamma', 2)
            rbf = rbf_kernel(self.gamma)
            self.kernel_fkt = rbf.calc
            self.b = kwargs.pop('b_init', -0.5)

        if self.kernel is 'linear':
            self.b = kwargs.pop('b_init', 0)
            self.kernel_fkt = lambda x1, x2: np.inner(x1, x2)
            self.mu = None

        self.C = kwargs.pop('C', 7)

        # learning rate
        self.learning_rate = kwargs.pop('learning_rate', 0.2)

        # conv_crit
        self.conv_crit = kwargs.pop('conv_crit', 10**-10)

        # n iterations
        self.n_iterations = kwargs.pop('n_iterations', 500)

        self._likelihood = None

        self.w = kwargs.pop('w_init', None)

        self.optimize_mu = False

    def train(self, dataset, mu = None):

        '''
        dataset:
        mu: {np.array} of shape (n_samples), elements only None or 0
            None -> always keeps in training
            0 -> can adjust mu to suppress influence on training

        '''

        # Convert labels to the format [1, -1]
        dataset = replace_labels(dataset, [1, -1])

        self._likelihood = [-10, -5]

        # X array, y list
        X, y = dataset.get_labeled_entries()

        if self.w is None:
            self.w = np.mean(X, axis = 0)
            #for _ in range(0,len(self.w),2):
            #    self.w[_] = -self.w[_]

        if mu is None:
            self.mu = np.array([None for _ in range(len(X))])
        else:
            self.mu = mu

        for mu in self.mu:
            if mu is not None:
                self.optimize_mu = True
                break

        # Fit till convergence or n_iterations fit calls
        while np.abs(self._likelihood[-1]-self._likelihood[-2]) > self.conv_crit:

            self._fit(X, y)
            self._likelihood.append(self._get_likelihood(X, y))

            if len(self._likelihood) > self.n_iterations:
                break

    def _fit(self, X, y):

        # linear
        if self.kernel == 'linear':
            self._fit_linear(X, y)

        # rbf
        if self.kernel == 'rbf':
            self._fit_rbf(X, y)

    def _fit_rbf(self, X, y):

        # Optimise w, b
        # First compute Kernel
        K = np.array(list(map(self.kernel_fkt, len(X)*[self.w], X)))

        # Calculate sigma's
        _sigma = np.zeros(len(X))
        for i in range(len(X)):
            if self.mu[i] is None:
                _sigma[i] = sigma(y[i]*(K[i]+self.b))
            else:
                _sigma[i] = sigma(y[i]*(K[i]+self.b+self.mu[i]))

        # Gradient
        grad = np.zeros(self.w.shape[0])
        for j in range(len(grad)):
            grad[j] = np.sum([(1-_sigma[i])*K[i]*y[i]*self.alpha*-2*self.gamma*(self.w[j]-X[i,j]) for i in range(len(X))])
        grad_b = np.sum([(1-_sigma[i])*y[i]*self.alpha for i in range(len(X))])
        grad = np.hstack((grad, grad_b))

        # Hessian
        hessian = np.zeros((X.shape[1]+1, X.shape[1]+1))
        for i in range(X.shape[1]):
            for j in range(i+1):
                hessian[i,j] = np.sum([4*self.gamma**2*K[k]*(self.w[i]-X[k,i])*(self.w[j]-X[k,j])*(1-_sigma[k])*y[k]*self.alpha*
                               (1-y[k]*self.alpha*K[k]*_sigma[k]) for k in range(len(X))])

                if j<i:
                    hessian[j,i] = hessian[i,j]

        for i in range(X.shape[1]):
            hessian[i,-1] = np.sum([2*self.alpha**2*self.gamma*_sigma[j]*(1-_sigma[j])*K[j]*(self.w[i]-X[j,i]) for j in range(len(X))])
            hessian[-1, i] = hessian[i,-1]

        hessian[-1, -1] = np.sum([-_sigma[i]*(1-_sigma[i])*self.alpha**2 for i in range(len(X))])

        # Compute ascent direction
        ascent_dir = np.dot(np.linalg.inv(hessian), grad)

        # Update w and b
        self.w -= self.learning_rate*ascent_dir[:-1]
        self.b -= self.learning_rate*ascent_dir[-1]

        ## Optimize mu
        self._optimize_mu(X, y)

    def _fit_linear(self, X, y):

        N = X.shape[0]
        n = X.shape[1]

        # First compute sigma
        _sigma = np.zeros(N)
        for i in range(N):
            if self.mu[i] is None:
                _sigma[i] = sigma(y[i]*(self.kernel_fkt(self.w, X[i])+self.b))
            else:
                _sigma[i] = sigma(y[i]*(self.kernel_fkt(self.w, X[i])+self.b+self.mu[i]))

        # Compute gradient
        grad = np.zeros(n)
        for j in range(n):
            grad[j] = np.sum([(1-_sigma[i])*self.alpha*y[i]*X[i,j] for i in range(N)])
        grad_b = np.sum([(1-_sigma[i])*y[i]*self.alpha for i in range(N)])
        grad = np.hstack((grad, grad_b))

        # Hessian
        hessian = np.zeros((n+1, n+1))
        for i in range(n):
            for j in range(i+1):
                hessian[i,j] = np.sum([-_sigma[k]*(1-_sigma[k])*self.alpha**2*X[k,i]*X[k,j] for k in range(N)])

                if j<i:
                    hessian[j,i] = hessian[i,j]

        for i in range(n):
            hessian[i,-1] = np.sum([-_sigma[j]*(1-_sigma[j])*self.alpha**2*X[j,i] for j in range(N)])
            hessian[-1, i] = hessian[i,-1]

        hessian[-1, -1] = np.sum([-_sigma[i]*(1-_sigma[i])*self.alpha**2 for i in range(N)])

        # Compute ascent direction
        ascent_dir = grad #np.dot(np.linalg.inv(hessian), grad)

        # Update w and b
        self.w += self.learning_rate*ascent_dir[:-1]
        self.b += self.learning_rate*ascent_dir[-1]

        ## Optimize mu
        self._optimize_mu(X, y)

    def _optimize_mu(self, X, y):
        if self.optimize_mu:
            b = []
            for i in range(len(X)):
                if self.mu[i] is not None:
                    b.append([i, y[i]*(self.kernel_fkt(self.w, X[i])+self.b)])
            b = np.asarray(b)

            # Sort
            b = sort_by_2nd(b, 'min')

            # Find n (see References)
            n = self._get_n(b[:,1])

            # update mu
            for idx in b[:n, 0].astype(int):
                self.mu[idx] = np.sum([y[idx]/n*y[a]*(self.kernel_fkt(self.w, X[a])+self.b) for a in b[:n, 0].astype(int)])+ \
                                N_a/n*y[idx]*self.C - (self.kernel_fkt(self.w, X[idx])+self.b)

            for idx in b[n:, 0].astype(int):
                self.mu[idx] = 0
        else:
            pass

    def _get_n(self, b):
        global N_a

        N_a = len(b)
        # Could do binary search here, much faster for sure

        for i in range(N_a):
            if (i+1)*b[i] - np.sum(b[:(i+1)]) > (N_a*self.C):
                break
            n = i+1

        return n


    def _get_likelihood(self, X, y):
        K = np.array(list(map(self.kernel_fkt, X.shape[0]*[self.w], X)))

        likelihood = 0
        for i in range(len(X)):
            if self.mu[i] is not None:
                likelihood += np.log(sigma(y[i]*(K[i]+self.b+self.mu[i])))
            else:
                likelihood += np.log(sigma(y[i]*(K[i]+self.b)))

        return likelihood


    def predict(self, X):
        K = np.array(list(map(self.kernel_fkt, X.shape[0]*[self.w], X)))
        _ = np.array([K[i]+self.b for i in range(len(X))])
        return np.sign(_)

    def score(self, dataset):
        # Convert to [1, -1] label format
        dataset = replace_labels(dataset, [1, -1])

        X, y = dataset.get_labeled_entries()

        pred = self.predict(X)
        return np.sum([pred[i]==y[i] for i in range(len(X))])/len(X)
