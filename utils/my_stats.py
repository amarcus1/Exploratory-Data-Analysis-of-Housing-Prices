# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 17:20:28 2022

@author: amarcus1
"""
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
 
def my_Cook_test(data, dependent, independent):
    """
    Computes the Cook distance between an independent and dependent variables
    
    Arguments:
        data -- datafile (Pandas Dataframe)
        dependent -- name of dependent variable (string) 
        independent -- name of dependent variable (string) 
    
    Returns:
        distance -- cooks distance (Pandas Series)
        p-value -- signifiance (Numpy NDarray)
    """    
    
    # fit the regression model using statsmodels library 
    f = dependent + ' ~ ' + independent
    model = ols(formula=f, data=data).fit()
    
    # calculate the cooks_distance - the OLSInfluence object contains multiple influence measurements
    cook_distance = OLSInfluence(model).cooks_distance
    (distance, p_value) = cook_distance
    
    return distance, p_value

def get_influencers(data, dependent, independent, threshold=-1):
    """
    Obtains a list of influential points based on the Cook's test
    
    Arguments:
        data -- datafile (Pandas Dataframe)
        dependent -- name of dependent variable (string) 
        independent -- name of dependent variable (string) 
        threshold -- set a threshold for influence (string) 
    
    Returns:
        a list of influential points
    """  

    distance, _ = my_Cook_test(data, dependent, independent)
 
    if threshold == -1:
        threshold = 4/data.shape[0]   
   
    # the observations with Cook's distances higher than the threshold value are labeled in the plot
    influencial_data = distance[distance > threshold]
    
    return list(influencial_data.index)
     
def my_Cook_plot(data, dependent, independent):
    """
    Create a Plot of Cook's Test
    
    Arguments:
        data -- datafile (Pandas Dataframe)
        dependent -- name of dependent variable (string) 
        independent -- name of dependent variable (string) 
    
    Returns:
        None
    """  
    
    distance, _ = my_Cook_test(data, dependent, independent)
    
    # scatter plot - x axis (independent variable height), y-axis (dependent variable weight), size and color of the marks according to its cook's distance
    plt.figure(figsize=(15,16))
    plt.subplot(2,1,1)
    ax = sns.scatterplot(data=data, x=dependent, y=independent, hue=distance, size=distance, sizes=(50, 200), edgecolor='black', linewidth=1)
    # ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # labels and title
    plt.xlabel(dependent, fontsize=14)
    plt.ylabel(independent, fontsize=14)
    plt.title('Cook\'s distance', fontsize=20)
    
    threshold = 4/data.shape[0]

    # stem plot - the x-axis represents the index of the observation and the y-axis its Cook's distance
    plt.subplot(2,1,2)
    plt.stem(distance, basefmt=" ")

    # horizontal line showing the threshold value
    plt.hlines(threshold, -2, data.shape[0], 'r')
    # the observations with Cook's distances higher than the threshold value are labeled in the plot
    influencial_data = distance[distance > threshold]
    
    for index, value in influencial_data.items():
        plt.text(index, value, str(index), fontsize=14)

    # ticks
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    # labels and title
    plt.xlabel('index', fontsize=14)
    plt.ylabel('Cook\'s distance', fontsize=14)
    plt.title('Cook\'s distance', fontsize=20)
    
    plt.savefig('./images/cookplot.png')

    plt.show()
    

def plot_krustal(examples, p_val, H_stat, df):
    """
    Create a Plot based on the Kruskal-Wallis Test
    
    Arguments:
        examples -- a list of features (list of strings)
        p_val -- p-values (Pandas Series) 
        H_stat -- H-Stat (Pandas Series) 
        df -- original data (Pandas Series) 
        
    Returns:
        None
    """  
    
    plt.figure(figsize = (12,7))

    # plot two examples of Boxplot
    for i in range(2):        
        df_idx=df.groupby([examples[i]])['SalePrice'].mean().sort_values(ascending=True).index       
        ax = plt.subplot2grid((2,2), (0,i))
        sns.boxplot(x=examples[i], y = 'SalePrice', data = df, ax = ax, order=df_idx, palette= sns.color_palette('winter'))
        ax.set_title(examples[i]+' - SalePrice Boxplot', color = 'k')
     #   ax.set_facecolor('pink')
        if i == 1:
            ax.set_yticks([])
            ax.set_ylabel('')        
    plt.subplots_adjust(top = 0.8)
    sns.despine(left = True)

    ax = plt.subplot2grid((2,2), (1,0), colspan = 2)
    p_val.sort_values().plot.bar(ax = ax, width= 1.5, color = 'b')
    ax.axhline(y = 0.05, color = 'red', linewidth = 2)
    ax.set_title('Kruskal Test to All Categorical Variables regard of Price', fontsize = 14, color = 'k')
    ax.set_ylabel('p-value')        
    plt.subplots_adjust(top = 0.8, hspace = 0.5)
    plt.suptitle('Kruskal-Walls Test Effect', fontsize = 14, color = 'k')
   
#    plt.show()
    for var in examples:
        a,b,c, *_  = df[var].unique()
        H_stat, p_val = stats.kruskal(df['SalePrice'].loc[df[var] == a], df['SalePrice'].loc[df[var] == b],df['SalePrice'].loc[df[var] == c])
        if p_val < 0.05: print(('P-value %.2f According to ' + str(var) + ', SalePrice distribution changed') % p_val)
        else: print(('P-value %.2f According to ' + str(var) + ", SalePrice distribution didn't changed") % p_val)
        
        
def my_kruskal(df, label):
    """
    Perform Kruskal-Wallis Test
    
    Arguments:
        df -- original data (Pandas Series) 
        examples -- target label
        
    Returns:
        p_val -- p-values (Pandas Series) 
        H_stat -- H-Stat (Pandas Series)         
    """  
    
    def kruskal_pval(x, y = df[label]):
        a,b,c,*_= x.unique()
        H_stat, p_val = stats.kruskal(y.loc[x == a], y.loc[x == b],y.loc[x == c])
        return H_stat, p_val
        
    cols =  df.columns[(df.apply(pd.Series.nunique) < 15) & (df.apply(pd.Series.nunique) > 2 )]
    kruskal_result = df[cols].apply(kruskal_pval)
    H_stat = kruskal_result.iloc[0]
    p_val = kruskal_result.iloc[1]
    p_val = p_val.dropna()
    
    return p_val, H_stat

from sklearn.feature_selection import mutual_info_regression  
from sklearn.feature_selection import SelectKBest, f_regression
from collections import defaultdict

def my_Kbest(X, y, method='mutual', k=10):
    """
    Performs Feature Selections 
    
    Arguments:
        X -- Features (Pandas Dataframe) 
        y -- label (Pandas Series) 
        method -- methods (e.g., mutual, ANOVA) (string)
        k -- number of features to select (int)
        
    Returns:
        summary -- summary of features selected or not (Pandas DataFrame) 
    """  
    
    methods = {
        'mutual': SelectKBest(mutual_info_regression, k=k),
        'anova': SelectKBest(f_regression, k=k),
    }
    selector = methods[method]
        
    _ = selector.fit_transform(X, y)  #Applying transformation to the training set
    #to get names of the selected features
    mask = selector.get_support()     # Output   array([False, False,  True,  True,  True, False ....])
 
    summary = defaultdict(list)
    for i, feature in enumerate(X.columns):
        summary['feature'].append(feature)
        summary['scores'].append(selector.scores_[i])
        summary['selected'].append(mask[i])

    summary = pd.DataFrame(summary)
    summary = summary.sort_values(by='scores', ascending=False)
    
    print(summary[summary.selected == True].head)
    return summary

from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
#from sklearn.feature_selection import SequentialFeatureSelector

def my_forward_selection(X, y, k=20):
    """
    Performs Forward Feature Selections 
    a future version may make the regression alogrithms a parameter
    
    Arguments:
        X -- Features (Pandas Dataframe) 
        y -- label (Pandas Series) 
        method -- methods (e.g., mutual, ANOVA) (string)
        k -- number of features to select (int)
        
    Returns:
        summary -- summary of features selected or not (Pandas DataFrame) 
    """  
    
    sfs = SequentialFeatureSelector(LinearRegression(),         # cv = k-fold cross validation, floating is another extension of SFS, not used here
               k_features=k, 
               forward=True, 
               floating=False,
               scoring='neg_mean_squared_error',
               cv=2)    
    sfs = sfs.fit(X, y)
        
    selected_features = X.columns[list(sfs.k_feature_idx_)]

    # print(sfs.k_score_)
    
    # Create a dataframe to summarize the forward selection results
    selected = [(feature in selected_features) for feature in X.columns] 
    summary = defaultdict(list)
    for i, feature in enumerate(X.columns):
        summary['feature'].append(feature)
        summary['selected'].append(selected[i])    
        
    summary = pd.DataFrame(summary)
    summary = summary.sort_values(by='selected', ascending=True)
    
    # print(summary[summary.selected == True].head)
    return summary        
    
from sklearn.feature_selection import RFE
    
def my_recursive_selection(X, y, k=20):       
    """
    Performs Recursive Feature Selections 
    a future version may make the regression alogrithms a parameter
    
    Arguments:
        X -- Features (Pandas Dataframe) 
        y -- label (Pandas Series) 
        method -- methods (e.g., mutual, ANOVA) (string)
        k -- number of features to select (int)
        
    Returns:
        summary -- summary of features selected or not (Pandas DataFrame) 
    """  
    
    lm = LinearRegression()
    rfe1 = RFE(lm, n_features_to_select=k)   # RFE with 20 features
    
    # Fit on train and test data with 20 features
    X_train_new = rfe1.fit_transform(X, y)
#    X_test_new = rfe1.transform(X_test)
    
    # # Print the boolean results
    # print(rfe1.support_)       # Output [False False False False  True False False False  True False False...]
    # print(rfe1.ranking_)       # Output [36 34 23 26  1 21 12 27  1 13 28  1 18 19 32 25  1 11  9  7  8 10 30 35...] 
    
    summary = defaultdict(list)
    for i, feature in enumerate(X.columns):
        summary['feature'].append(feature)
        summary['rank'].append(rfe1.ranking_[i])
        summary['selected'].append(rfe1.support_[i])    
        
    summary = pd.DataFrame(summary)
    summary = summary.sort_values(by='rank', ascending=True)
    
    # print(summary[summary.selected == True].head)
    return summary    
    
def feature_selection_summary(df, label, k=20):
    
    corr = df.corr()['SalePrice'].abs().sort_index()
    corr = corr.drop('SalePrice')
    
    X = df.drop('SalePrice', axis=1)
    y = df['SalePrice']
    
    mutual = my_Kbest(X, y, 'mutual', k=k).sort_values(by='feature', ascending=True)
    anova  = my_Kbest(X, y, 'anova', k=k).sort_values(by='feature', ascending=True)
    forward = my_forward_selection(X, y, k=k).sort_values(by='feature', ascending=True)
    recursive = my_recursive_selection(X, y, k=k).sort_values(by='feature', ascending=True)

    summary = mutual[['feature']].copy()
    
    summary['Mutual'] = mutual['scores']
    summary['ANOVA_F'] = anova['scores']
    summary['Forward'] = forward['selected']
    summary['Recursive'] = recursive['selected']
        
    summary.insert(1, 'Cov(SalePrice)', corr.values)

    summary = summary.sort_values(by='Cov(SalePrice)', ascending=False)

    from IPython.display import display

    display(summary.style.format({
        'Cov(SalePrice)': '{:.2f}', 
        'Mutual': '{:.2f}',                     
        'ANOVA_F': '{:.0f}'
        }).hide(axis='index')
        )
    return summary    

