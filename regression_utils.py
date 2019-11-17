import numpy as np
from sklearn.metrics import r2_score


def helloworld():
    #print("Hello, world!")
    print("uncomment me")
    return


def simple_linear_regression(x, y):
    """
    Implement simple linear regression below
    y: list of dependent variables
    x: list of independent variables
    return: b1- slope, b0- intercept
    """
    y_mean = np.mean(y)
    x_mean = np.mean(x)
    
    b1 = sum(\
             (x[i][0] - x_mean) * (y[i] - y_mean) \
             for i in range(len(x))
            ) / sum((x[i][0] - x_mean)**2 for i in range(len(x)))
    
    b0 = y_mean - b1*x_mean

    return b1, b0


def multiple_regression(x, y):
    """
    x: np array of shape (n, p) where n is the number of samples
    and p is the number of features.
    y: np array of shape (n, ) where n is the number of samples
    return b: np array of shape (n, )
    """
    # Need to add a column as the first column to get b0.
    x.insert(loc=0, column='B0', value=np.ones(shape=x.shape[0]).reshape(-1,1))
    x_transpose = x.transpose()
    t = x_transpose.dot(x)
    _inverse = np.linalg.pinv(t.values)
    b = _inverse.dot(x_transpose).dot(y)
    return b


def predict(x, b):
    """
    This function would allow us to perform predictions.
    As we are performing linear regression we need to consider the b0 value as well.
    This can be achieved by using the beta values and the first beta value which is b0
    is what we would use here.
    """
    if len(b) < 2:
        return
    b0, b_coeff = b[0], b[1:]
        
    # Multiply all the beta_coeffecients with their respective columns.
    x *= b_coeff

    # Compute the sum of each row  and also add the b0 which we had segregated earlier.
    yhat = x.sum(axis=1)
    yhat += b0
    return yhat


def calculate_r2(y, yhat):
    # y: np array of shape (n,) where n is the number of samples
    # yhat: np array of shape (n,) where n is the number of samples
    # yhat is calculated by predict()

    # calculate the residual sum of squares (rss) and total sum of squares (tss)
    rss = sum((y[i] - yhat[i])**2 for i in range(len(y)))
    y_mean = np.mean(y)
    tss = sum((y[i] - y_mean)**2 for i in range(len(y)))
    r2 = 1.0 - rss/tss
    return r2


def calculate_adjusted_r2(y, yhat, num_of_features):
    r2 = calculate_r2(y, yhat)
    num_samples = len(y)
    
    return (1 - ((1-r2)*(num_samples-1)/(num_samples-1-num_of_features)))

def check_r2(y, yhat):
    return np.allclose(calculate_r2(y, yhat), r2_score(y, yhat))
