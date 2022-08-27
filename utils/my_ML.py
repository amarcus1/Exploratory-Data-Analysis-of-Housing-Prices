# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 09:28:32 2022

@author: amarcus1
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVR


from sklearn.linear_model import LinearRegression

class MyRegressors:
    """
    Performs regression analysis using common regressors
    
    """  
    
    def __init__(self, data):
        self.data = data
        self.LRfit()
        self.SVMfit()
        self.RFRfit()    
        
    def LRfit(self):   
        self.LR_reg = {}                    
        for k in range(self.data.k):
            X_train, y_train = self.data.get_k_train(k)
        
            self.LR_reg[k] = LinearRegression()
            self.LR_reg[k].fit(X_train, y_train)
                          
    def LRpredict(self, X): # predict should take X_test provided from outside       
        X = self.data.transform(X)        
        return self.LR_reg.predict(X)     
    
    def LRmetrics(self):
        scores = []
        for k in range(self.data.k):
            X_valid, y_valid = self.data.get_k_validation(k)
            y_pred = self.LR_reg[k].predict(X_valid)
            scores.append(self.evaluation(y_valid, y_pred))
        scores = np.array(scores)
        return list(np.mean(scores, axis=0))

    def LRresults(self):
        print('__________________________________')
        print('Linear Regression Cross Validation')
        self.print_evaluate(self.LRmetrics)

    def evaluation(self, true, predicted):
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mean_squared_error(true, predicted))
        r2_square = r2_score(true, predicted)
        rho = ccc(true, predicted)

        return [mae, mse, rmse, r2_square, rho]

    def print_evaluate(self, fn):  
        r = fn()
        
        print('MAE: {:.0f}'.format(r[0]))
        print('MSE: {:.0f}'.format(r[1]))
        print('RMSE: {:.0f}'.format(r[2]))
        print('R2 Square: {:.3f}'.format(r[3]))
        print('CCC: {:.3f}'.format(r[4]))   
        print('__________________________________')

    def SVMfit(self):
        self.SVM_reg = {}                    
        for k in range(self.data.k):
            X_train, y_train = self.data.get_k_train(k)
        
            self.SVM_reg[k] = SVR(C=1000000, epsilon=0.001, kernel='rbf') 
            self.SVM_reg[k].fit(X_train, y_train)
            
    def SVMpredict(self):
        return self.SVM_reg.predict(self.X_test)
    
    def SVMmetrics(self):
        scores = []
        for k in range(self.data.k):
            X_valid, y_valid = self.data.get_k_validation(k)
            y_pred = self.SVM_reg[k].predict(X_valid)
            scores.append(self.evaluation(y_valid, y_pred))
        scores = np.array(scores)
        return list(np.mean(scores, axis=0))
            
    def SVMresults(self):
        print('__________________________________')
        print('SVM Regression Cross Validation')
        self.print_evaluate(self.SVMmetrics)

    def performance_summary(self):     
        cols = ['Model','MAE','MSE','RMSE','R-squared','CCC']
        d = {}
        for col in cols:
            d[col] = []
        df = pd.DataFrame(d)
        
        self.LRfit()
        df.loc[0] = ['Linear Regression'] + self.LRmetrics()
        self.SVMfit()
        df.loc[1] = ['SVM Regression'] + self.SVMmetrics()
        self.RFRfit()
        df.loc[2] = ['Random Forest'] + self.RFRmetrics()
        
        print('Cross Validation (k=' + str(self.data.k) + ') Results for Regressors ')
        print(df)

    def RFRfit(self, max_depth=12):    
        self.RFR_reg = {}                    
        for k in range(self.data.k):
            X_train, y_train = self.data.get_k_train(k)
        
            self.RFR_reg[k] = RandomForestRegressor(n_estimators = 500, max_depth=max_depth)
            self.RFR_reg[k].fit(X_train, y_train)
    
    def RFRpredict(self):
        return self.RFR_reg.predict(self.X_test)
        
    def RFRmetrics(self):
        scores = []
        for k in range(self.data.k):
            X_valid, y_valid = self.data.get_k_validation(k)
            y_pred = self.RFR_reg[k].predict(X_valid)
            scores.append(self.evaluation(y_valid, y_pred))
        scores = np.array(scores)
        return list(np.mean(scores, axis=0))
        
    def RFRresults(self):
        print('__________________________________')
        print('Random Forest Cross Validation')
        self.print_evaluate(self.RFRmetrics)
                
    def RFR_Feature_Selection(self):
            
        result = permutation_importance(
            self.RFR_reg[0], self.data.X, self.data.y, n_repeats=10, random_state=42, n_jobs=2
        )
            
        sorted_importances_idx = result.importances_mean.argsort()
        importances = pd.DataFrame(
            result.importances[sorted_importances_idx].T,
            columns=self.data.feature_names[sorted_importances_idx],
        )
        
        ax = importances.iloc[:, -15:-1].plot.box(vert=False, whis=10, figsize=(12,5))
        ax.set_title("Permutation Importances of Numerical Data (test set)")
        ax.axvline(x=0, color="b", linestyle="--")
        ax.set_xlabel("Decrease in accuracy score")
        ax.figure.tight_layout()
        plt.show()
        
        return importances

    def RFRplot(self):
        X_valid, y_valid = self.data.get_k_validation(0)        
        ypred = self.RFR_reg[0].predict(X_valid)
        sns.scatterplot(x=y_valid, y=ypred)
        plt.xlabel('Actual', fontsize=14)
        plt.ylabel('Prediction', fontsize=14)
        plt.title('Sales Price Model-Actual Comparison', fontsize=20)
        plt.show()
  

def ccc(x,y):
    """
    Calculates the concordance correlation coefficient (CCC)
    
    Arguments:
        x -- model prediction    (n, 1) numpy array 
        k -- actual observations (n, 1) numpy array 
    
    Returns:
        ccc -- (n, 1) numpy array 
    """    
        
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

# def evaluate(true, predicted):
#     mae = mean_absolute_error(true, predicted)
#     mse = mean_squared_error(true, predicted)
#     rmse = np.sqrt(mean_squared_error(true, predicted))
#     r2_square = r2_score(true, predicted)
#     return mae, mse, rmse, r2_square

