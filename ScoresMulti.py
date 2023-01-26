import numpy as np

def ucf_score(y_test, y_pred_test, delta = 0.5):
    
    y_diff = np.square(y_pred_test - y_test).sum(axis=1)
    y_diff = np.sqrt(y_diff)
    
    y_ucf = np.empty((y_diff.shape))
    y_ucf[np.where(y_diff <= delta/2)] = 0
    y_ucf[np.where(y_diff > delta/2)] = 1
    ucf = float(y_ucf.mean())
    
    return ucf
  


def aRRMSE(y_test, y_pred_test):
    
    result = 0
    for i in range(y_test.shape[1]):
        up = ((y_test[:,i] - y_pred_test[:,i])**2).sum()
        down = ((y_test[:,i] - y_test[:,i].mean())**2).sum()
        result += np.sqrt(up/down)
    result /= y_test.shape[1]
    
    return result

    
def correntropy(y_test, y_pred_test, h=0.5):
    def phi(y, h=0.5):
        return np.exp(-(y**2)/(h**2))
    
    y_corr = h**2 * (1 - phi(y_pred_test - y_test,h))
    return y_corr.mean().mean()

