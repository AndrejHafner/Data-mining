import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import math


def load(name):
    """ 
    Odpri datoteko. Vrni matriko primerov (stolpci so znacilke) 
    in vektor razredov.
    """
    data = np.loadtxt(name)
    X, y = data[:, :-1], data[:, -1].astype(np.int)
    return X, y

def regularization(lambda_,thetas,type="l1"):
    if type == "l1":
        return lambda_ * np.sum(np.abs(thetas))
    elif type =="l2":
        return lambda_ * np.sum(np.square(thetas))
    else:
        return 0

def h(x, theta):
    """ 
    Napovej verjetnost za razred 1 glede na podan primer (vektor vrednosti
    znacilk) in vektor napovednih koeficientov theta.
    """
    return 1 / (1 + np.exp(-(x.dot(theta))))


def cost(theta, X, y, lambda_):
    """
    Vrednost cenilne funkcije.
    """
    m = len(y)
    A = np.array([h(X[i],theta) for i in range(m)])
    # Vectorized form
    # Fires divide by zero error -> actually error because of trying to calculate log(0)
    res = (-1/m) * np.sum((y.transpose() @ np.log(A)) + ((1-y).transpose() @ np.log(1-A))) + lambda_*np.sum(np.square(theta))/(2*m)
    return res


def grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije. Vrne numpyev vektor v velikosti vektorja theta.
    """


    m = len(y)
    grad = np.zeros(theta.shape)
    hx = np.array([h(X[i], theta) for i in range(m)])

    for i in range(len(theta)):
        xt = X[:,i]
        grad[i] = (1/m) * np.sum((hx-y)*xt) + ((lambda_ * theta[i]) / m)

    return grad


def num_grad(theta, X, y, lambda_):
    """
    Odvod cenilne funkcije izracunan numericno.
    Vrne numpyev vektor v velikosti vektorja theta.
    Za racunanje gradienta numericno uporabite funkcijo cost.
    """
    eps = 0.0001
    grad = np.zeros(theta.shape)
    for i in range(len(theta)):
        theta_curr = theta[i]
        theta[i] = theta_curr + eps
        c1 = cost(theta,X,y,lambda_)

        theta[i] = theta_curr - eps
        c2 = cost(theta,X,y,lambda_)
        theta[i] = theta_curr

        grad[i] = (c1 - c2) / (2*eps)

    return np.array(grad)


class LogRegClassifier(object):

    def __init__(self, th):
        self.th = th

    def __call__(self, x):
        """
        Napovej razred za vektor vrednosti znacilk. Vrni
        seznam [ verjetnost_razreda_0, verjetnost_razreda_1 ].
        """
        x = np.hstack(([1.], x))
        p1 = h(x, self.th)  # verjetno razreda 1
        return [1-p1, p1]


class LogRegLearner(object):

    def __init__(self, lambda_=0.0):
        self.lambda_ = lambda_

    def __call__(self, X, y):
        """
        Zgradi napovedni model za ucne podatke X z razredi y.
        """
        X = np.hstack((np.ones((len(X),1)), X))

        # optimizacija
        theta = fmin_l_bfgs_b(
            cost,
            x0=np.zeros(X.shape[1]),
            args=(X, y, self.lambda_),
            fprime=grad)[0]

        return LogRegClassifier(theta)


def test_learning(learner, X, y):
    """ vrne napovedi za iste primere, kot so bili uporabljeni pri učenju.
    To je napačen način ocenjevanja uspešnosti!

    Primer klica:
        res = test_learning(LogRegLearner(lambda_=0.0), X, y)
    """
    c = learner(X,y)
    results = [c(x) for x in X]
    return results


def test_cv(learner, X, y, k=5):
    predictions = []

    # Shuffle
    seed = 42
    rng = range(0,X.shape[0])

    np.random.seed(seed)
    shuffle_idx = list(rng)
    np.random.shuffle(shuffle_idx)
    X_shuffle = np.array([X[k] for k in shuffle_idx])
    y_shuffle = np.array([y[k] for k in shuffle_idx])

    for i in range(1,k+1):
        X_train, X_test, Y_train, Y_test = kfold(X_shuffle,y_shuffle,i,k)
        classifier = learner(X_train,Y_train)
        predictions = predictions + [classifier(row) for row in X_test]

    return [predictions[shuffle_idx.index(j)] for j in rng]




def CA(real, predictions):
    return sum([1 if (predictions[i][0] > 0.5 and real[i] == 0) or (predictions[i][1] > 0.5 and real[i] == 1) else 0 for i in range(len(real))]) / len(real)

def AUC(real, predictions):
    pred = [x[1] for x in predictions]
    cnt = 0
    sum = 0
    rng = range(len(real))
    for i in rng:
        for j in range(i+1,len(real)):
            if real[i] != real[j]:
                cnt += 1

                if pred[i] == pred[j]:
                    sum += 0.5
                elif pred[i] < pred[j]:
                    sum += 1

    return sum / cnt

def kfold(X,y,i,k):
    n = len(X)
    indexes_to_keep = set(range(X.shape[0])) - set(range(n*(i-1)//k,(n*i//k)))
    train = list(indexes_to_keep)
    test = list(range(n*(i-1)//k,n*i//k))
    X_train = X[train]
    X_test = X[test]
    Y_train = y[train]
    Y_test = y[test]

    return X_train,X_test,Y_train,Y_test


if __name__ == "__main__":
    # Primer uporabe

    learner = LogRegLearner(lambda_=0.0)

    X, y = load('reg.data')

    print(test_cv(learner,X,y,5))


    classifier = learner(X, y) # dobimo model
    print(test_learning(learner,X,y))
    #napoved = classifier(X[0])  # napoved za prvi primer
    #print(napoved)

