import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from word2number import w2n
import math

def gradient_descent(x,y):
    m_curr = 0
    b_curr = 0
    iterations = 100000
    n = len(x)
    learning_rate = 0.0002
    cost = 0
    
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = -(2/n)*sum(x*(y-y_predicted))
        bd = -(2/n)*sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print('m {}, b {}, cost {} iteration {}'.format(m_curr,b_curr,cost,i))
        
df = pd.read_csv('test_scores.csv')
x = np.array(df.math)
y = np.array(df.cs)

gradient_descent(x,y)
plt.show()
