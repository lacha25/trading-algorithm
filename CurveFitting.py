import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit



def linear_regression(time, price):
    model = LinearRegression()
    time = time.reshape(-1, 1)  
    model.fit(time, price)
    predictions = model.predict(time)
    return model.coef_[0], model.intercept_, predictions

