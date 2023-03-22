import numpy as np

def adapt_stepsize(sigma, tau, theta, n):
    if np.sqrt(theta*sigma*tau) >= n:
        sigma = n
    elif np.sqrt(sigma * tau) >= n and n > np.sqrt(theta*sigma*tau):
        sigma = np.sqrt(theta * sigma * tau)
    else:
        sigma = np.sqrt(sigma * tau)
    return sigma