#saitekikatrue

import numpy as np
from scipy.optimize import minimize
import pandas as pd
from sklearn.metrics import mean_squared_error,mean_absolute_error,mean_absolute_percentage_error
import math
#アンケート数値から重み付けの値を決定

def ftrue(x):
    #アンケート結果から求まった数値
    answear = np.array([72.5, 83, 71, 65, 38, 27.5, 39, 59, 55.5, 73])
    Valence = np.array([4.2, 4.8, 4.2, 4, 2.5, 1.9, 2.8, 3.6, 3.4, 3.9])
    Arousal = np.array([4.05, 4.5, 3.05, 2.6, 1.85, 1.25, 4.85, 3.75, 1.95, 3])
    a=x[0]
    b=x[1]
    score = a*(25*(Valence-1))+b*(25*(Arousal-1))
    print(a)
    print(b)
    print(score)
    RMSE = mean_squared_error(answear, score, squared=False)
    MSE = mean_squared_error(answear, score, squared=True)
    MAE = mean_absolute_error(answear, score)
    MAPE = mean_absolute_percentage_error(answear, score)
    return MAE

def con(x):
    a=x[0]
    b=x[1]
    return a+b-1
    
x0=[0.5,0.5]

#制約条件式：(x>0,y>0,x+y=1)
consall = (
    {'type': 'eq', 'fun':con},
)

#値域，定義域
bound=[(0,1),(0,1)]

result = minimize(ftrue, x0=x0, bounds=bound, constraints=consall, method="SLSQP")
print(result)