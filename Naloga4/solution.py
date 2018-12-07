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
    # ... dopolnite (naloga 3)
    pass


def CA(real, predictions):
    # ... dopolnite (naloga 3)
    pass


def AUC(real, predictions):
    # ... dopolnite (dodatna naloga)
    pass


if __name__ == "__main__":
    # Primer uporabe


    X, y = load('reg.data')

    learner = LogRegLearner(lambda_=0.0)
    classifier = learner(X, y) # dobimo model
    print(test_learning(learner,X,y))
    #napoved = classifier(X[0])  # napoved za prvi primer
    #print(napoved)

