#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np    # to use numpy n-dim array for transition matrix
from scipy.integrate import quad    # to calculate integration 
from scipy.stats import norm    # to calculate cdf of standard normal in BS


# In[2]:


#Proposition 1: stock price
def B0(l_f, l_s, lamb, r, g):
    numerator = 1.0 + l_f / (r - g + lamb)
    denominator = l_s + l_f + r - g - (1. * lamb * l_f) / (r - g + lamb)
    return 1. * numerator / denominator

def B1(l_f, l_s, lamb, r):
    numerator = 1.0 * l_s
    denominator = l_s + l_f + r - (1. * lamb * l_f) / (r + lamb)
    return 1. * numberator / denominator

def B2(l_f, l_s, lamb, q_a, phi):
    numerator = l_s * phi
    denominator = l_s + l_f + q_a - (lamb * l_f) / (q_a + lamb)
    return 1. * numerator / denominator

def A0(l_f, l_s, lamb, r, g):
    temp = B0(l_f, l_s, lamb, r, g)
    return (1. + lamb * temp) / (r - g + lamb)

def A1(l_f, l_s, lamb, r):
    temp = B1(l_f, l_s, lamb, r)
    return (1. * lamb * temp) / (r + lamb)

def A2(l_f, l_s, lamb, q_a, phi):
    temp = B2(l_f, l_s, lamb, q_a, phi)
    return (1. * lamb * temp) / (q_a + lamb)
    
def stock_price (A0, A1, A2, B0, B1, B2, Dt, V, Pa, state):
    """
    The function is to calculus target company price, 
    which is a probability weighted price of 
    target company, offer price and acquired company
    """
    if state == "Pretarget":
        return A0 * Dt + A1 * V + A2 * Pa
    elif state == "Target":
        return B0 * Dt + B1 * V + B2 * Pa
    else: 
        print("wrong state")


# In[4]:


# Proposition 2: Transition Probabilities
# This part is to construct the continuous Markov chain of risk-neutral 
# transition probabilities between 3 states: Preacquired, Target and Acquired
def transition_matrix(l_f, l_s, lamb, tau):
    """
    This is to calcluate the Markov transition probabilities matrix for
    3 state: Preacquired(P), Target(T), and Acquired(A)
    ===================================================================
    Input: 
    ===================================================================
    Output: 
    Numpy 3 by 3 ndim-array
    """
    v1 = .5 * (-lamb - l_f - l_s - np.sqrt( (lamb + l_f + l_s) ** 2 - 4 * lamb * l_s )) 
    v2 = .5 * (-lamb - l_f - l_s + np.sqrt( (lamb + l_f + l_s) ** 2 - 4 * lamb * l_s ))
    mul = (lamb / (v2 - v1))
    q11 = mul * ( (v2/lamb + 1) * np.exp(v1 * tau) - (v1/lamb + 1) * np.exp(v2 * tau))
    q12 = mul * (np.exp(v2 * tau) - np.exp(v1 * tau))
    q21 = mul * ((v2/lamb + 1) * (v1/lamb + 1) * (np.exp(v1 * tau) - np.exp(v2 * tau)))
    q22 = mul * ((v2/lamb + 1) * np.exp(v2 * tau) - (v1/lamb + 1) * np.exp(v1 * tau) )
    q13 = 1 - q11 - q12
    q23 = 1 - q21 - q22
    return np.array([[q11, q12, q13], [q21, q22, q23], [0, 0, 1.]])


# In[6]:


# Proposition 3 Option Prices:
# NOTE that the below part only implement cash deals
def BS_put(S, K, tau, r, q, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * (sigma ** 2)) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    price = - S * np.exp(-q * tau) * norm.cdf( - d1) + K * np.exp(-r * tau) * norm.cdf(-d2)
    return price

def F_P_CashDeal(A0, A1, D0, K, V, tau, r, g, sigma):
    return BS_put(A0 * D0, K - A1a, tau, r, r - g, sigma)

def F_T_CashDeal(B0, B1, D0, K, V, tau, r, g, sigma):
    return BS_put(B0 * D0, K - B1 * V, tau, r, r - g, sigma)

def F_A_CashDeal(r, tau, K, V):
    return np.exp(-r * tau) * max(K - V, 0)

def Euro_put (K, tau, transition_matrix, F_P, F_T, F_A, state):
    """
    This function is to output the European put option price
    =========================================================
    Input:
    1. K: strike price
    2. tau: time to maturity, in unit year
    3. transition_matrix: calculated by function transition_matrix()
    4. state: string, either "Pretarget" or "Target"
    =========================================================
    the price of European put option in 2 states: Pretarget(P), Target(T)
    """
    
    if state == "Pretarget":
        F = transition_matrix[0][0] * F_P + transition_matrix[0][1] * F_T + transition_matrix[0][2] * F_A
    if state == "Target":
        F = transition_matrix[1][0] * F_P + transition_matrix[1][1] * F_T + transition_matrix[1][2]
    else:
        print("Wrong state")


# In[4]:


def rn_prob_success(l_s, l_f):
    """
    This function is to calculate the risk neutral probability of deal success.
    ==========================================================================
    Input:
    1. l_s: lambda_s is the intensity of deal success
    2. l_f: lambda_f is the intensity of deal fail
    ==========================================================================
    Output:
    risk-neutral probability of deal success
    """
    return 1.0 * l_s/(l_s + l_f)

def obj_prob_success(p_q, kappa_f, kappa_s):
    """
    This function is to calculate the objective probability of deal success.
    aka. formula (14) in  Tassel (2016)
    ======================================================================
    Input:
    1. p_q: risk-neutral probability of deal success
    2. kappa_s: market price of deal success
    3. kappa_f: market price of deal failure
    ======================================================================
    Output
    objective probability of deal success
    """
    return p_q + p_q * (1 - p_q) * (kappa_f - kappa_s)


# In[8]:


# Estimation:
r, g, lamb, tau = 0.03, 0.01, 0.01, 0.33


# In[ ]:


#Proposition 4: Merger Arb Risk Premium:


# In[6]:


#test
rn_prob = rn_prob_success(2, 1)
print(rn_prob)
obj_prob = obj_prob_success(rn_prob, 0.2, 0.1)
print(obj_prob)


# In[23]:


def add(a, b):
    return a + b
def add_again(a, b):
    temp = add(a, b)
    return a + b + temp


# In[24]:


add(1, 2)


# In[25]:


add_again(1, 2)


# In[31]:


A0(l_f = 1, l_s = 2, lamb = 1, r = .03, g = .01)


# In[41]:


np.exp(1)


# In[5]:


# test proposition 2 matrix function
test_matrix = transition_matrix(l_f = 1, l_s = 2, lamb = 1, tau = 0.3)


# In[6]:


test_matrix


# In[7]:


test_matrix[1,1]


# In[37]:




I = quad(lambda x: x**2, 0, 1)
I[0]


# In[19]:


from scipy.stats import norm


# In[22]:


norm.cdf(1) - norm.cdf(-1)


# In[23]:


np.exp(1)


# In[26]:


np.log(np.exp(1))


# In[28]:


def BS_put(S, K, tau, r, q, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * (sigma ** 2)) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    price = - S * np.exp(-q * tau) * norm.cdf( - d1) + K * np.exp(-r * tau) * norm.cdf(-d2)
    return price


# In[29]:


BS_put(100, 100, 1, 0, 0, 0.2)


# In[30]:


def approx_ATM (S, K, tau, r, q, sigma):
    return 0.4 * sigma * S * np.sqrt(tau)


# In[31]:


approx_ATM(100, 100, 1, 0, 0, 0.2)


# In[ ]:




