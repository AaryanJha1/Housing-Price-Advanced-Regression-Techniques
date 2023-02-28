import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor  
from sklearn.impute import KNNImputer
from scipy import stats
from scipy.stats import norm, skew
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold



# =============================================================================
def main():
    # Read the original data files
    trainDF = pd.read_csv("data/train.csv")
    testDF = pd.read_csv("data/test.csv")
            
    #demonstrateHelpers(trainDF)

    trainInput, testInput, trainOutput, testIDs, predictors = transformData(trainDF, testDF)
    
    doExperiment(trainInput, trainOutput, predictors)
    
    doExperiment1(trainInput, trainOutput, predictors)
    
    doExperiment2(trainInput, trainOutput, predictors)
    
    doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors)

    
# ===============================================================================

'''
Does k-fold CV on the Kaggle training set using LinearRegression.
(You might review the discussion in hw06 about the so-called "Kaggle training set"
versus other sets.)
'''
def doExperiment(trainInput, trainOutput, predictors):
    alg = LinearRegression()
    alg.fit(trainInput.loc[:,predictors], trainOutput)
    cvMeanScore = model_selection.cross_val_score(alg, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score:", cvMeanScore)

def doExperiment1(trainInput, trainOutput, predictors):
    alg1 = Ridge()
    alg1.fit(trainInput.loc[:,predictors], trainOutput)
    cvMeanScore = model_selection.cross_val_score(alg1, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    print("CV Average Score:", cvMeanScore)

def doExperiment2(trainInput, trainOutput, predictors):
    alg3 = LinearRegression()
    cvMeanScore = model_selection.cross_val_score(alg3, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2', n_jobs=-1).mean()
    
    # We added GradientBoostingRegressor algorithm within the doExperiment Function
    
    alg2 = GradientBoostingRegressor()
    alg2.fit(trainInput.loc[:,predictors], trainOutput)
    cvScores = model_selection.cross_val_score(alg2, trainInput.loc[:, predictors], trainOutput, cv=10, scoring='r2').mean()
    print("CV Average Score: ", cvScores)




# ===============================================================================
'''
Runs the algorithm on the testing set and writes the results to a csv file.
'''
def doKaggleTest(trainInput, testInput, trainOutput, testIDs, predictors):
    alg = LinearRegression()

    # Train the algorithm using all the training data
    alg.fit(trainInput.loc[:, predictors], trainOutput)

    # Make predictions on the test set.
    predictions = alg.predict(testInput.loc[:, predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "Id": testIDs,
        "SalePrice": predictions
    })

    # Prepare CSV
    submission.to_csv('data/testResults.csv', index=False)
    # Now, this .csv file can be uploaded to Kaggle

# ============================================================================
# Data cleaning - conversion, normalization

'''
Pre-processing code will go in this function (and helper functions you call from here).
'''

def preprocessing(trainDF, testDF):
    #See what is in the file
    print(trainDF.head())
    print(testDF.head())
    
    print("----------------------------------------------------------------------------")

    print(trainDF.describe().transpose())
    print(testDF.describe().transpose())
    
    print("----------------------------------------------------------------------------")    
        
    sns.set(rc={'figure.figsize':(20,15)})
    sns.displot(trainDF['SalePrice'],kde=True,bins=20) 
    sns.kdeplot(trainDF['SalePrice'])
    
    #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    trainDF["SalePrice"] = np.log1p(trainDF["SalePrice"])

    #Check the new distribution 
    sns.distplot(trainDF['SalePrice'] , fit=norm);

    # Get the fitted parameters used by the function
    (mu, sigma) = norm.fit(trainDF['SalePrice'])
    print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))

    #Now plot the distribution
    plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
    plt.ylabel('Frequency')
    plt.title('SalePrice distribution')

    #Get also the QQ-plot
    fig = plt.figure()
    res = stats.probplot(trainDF['SalePrice'], plot=plt)
    plt.show()
    
    print("----------------------------------------------------------------------------")
            
    print(getAttrsWithMissingValues(trainDF))
    print(trainDF[getAttrsWithMissingValues(trainDF)].isnull().sum())
    
    trainDF.drop(['PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu', 'Alley'], axis=1, inplace=True)
    
    
    handle_non_numerical_data(trainDF)
    
    imputer = KNNImputer(n_neighbors=5)
    trainDF['LotFrontage'] = imputer.fit_transform(trainDF[['LotFrontage']])
    trainDF['MasVnrArea'] = imputer.fit_transform(trainDF[['MasVnrArea']])
    trainDF['GarageYrBlt'] = imputer.fit_transform(trainDF[['GarageYrBlt']])


    print("----------------------------------------------------------------------------")
    
    
    print(getAttrsWithMissingValues(testDF))
    print(testDF[getAttrsWithMissingValues(testDF)].isnull().sum())
    
    testDF.drop(['PoolQC','Fence','MiscFeature','FireplaceQu','Alley'],axis=1,inplace=True)

    handle_non_numerical_data(testDF)    
    
    imputer = KNNImputer(n_neighbors=5)
    testDF['LotFrontage'] = imputer.fit_transform(testDF[['LotFrontage']])
    testDF['MasVnrArea'] = imputer.fit_transform(testDF[['MasVnrArea']])
    testDF['BsmtFinSF1'] = imputer.fit_transform(testDF[['BsmtFinSF1']])
    testDF['BsmtFinSF2'] = imputer.fit_transform(testDF[['BsmtFinSF2']])
    testDF['BsmtUnfSF'] = imputer.fit_transform(testDF[['BsmtUnfSF']])
    testDF['TotalBsmtSF'] = imputer.fit_transform(testDF[['TotalBsmtSF']])
    testDF['BsmtFullBath'] = imputer.fit_transform(testDF[['BsmtFullBath']])
    testDF['BsmtHalfBath'] = imputer.fit_transform(testDF[['BsmtHalfBath']])
    testDF['GarageYrBlt'] = imputer.fit_transform(testDF[['GarageYrBlt']])
    testDF['GarageCars'] = imputer.fit_transform(testDF[['GarageCars']])
    testDF['GarageArea'] = imputer.fit_transform(testDF[['GarageArea']])
 
    print("----------------------------------------------------------------------------")
    
    correlation= abs(trainDF.corr())    #abs to find the magnitude of the square to include negative values.
                                           #corr is inbuilt function to find correlation between diff atributes.
    plt.figure(figsize = (80, 25))
    correlation['SalePrice'].sort_values(ascending = False).plot(kind = 'bar')
    plt.title('The correlation graph with SalesPrice', fontsize= 16);
    plt.show() 
    
    k = 10
    max_corr = correlation.nlargest(k, 'SalePrice')['SalePrice'].index
    min_corr = correlation.nsmallest(k, 'SalePrice')['SalePrice'].index
    print(max_corr)
    print(min_corr)

    print("----------------------------------------------------------------------------")
    
    sns.set()
    cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
    sns.pairplot(trainDF[cols], size = 2.5)
    plt.show();

    var = 'GrLivArea'
    data = pd.concat([trainDF['SalePrice'], trainDF['GrLivArea']], axis=1)
    data.plot.scatter(x='GrLivArea', y='SalePrice', ylim=(0,800000));
    
    trainDF.sort_values(by = 'GrLivArea', ascending = False)[:2]
    trainDF = trainDF.drop(trainDF[trainDF['Id'] == 1299].index)
    trainDF = trainDF.drop(trainDF[trainDF['Id'] == 524].index)
    
    print("----------------------------------------------------------------------------")

    trainDF['SalePrice'] = np.log(trainDF['SalePrice'])
    trainDF['GrLivArea'] = np.log(trainDF['GrLivArea'])
    
    trainDF['HasBsmt'] = pd.Series(len(trainDF['TotalBsmtSF']), index=trainDF.index)
    trainDF['HasBsmt'] = 0 
    trainDF.loc[trainDF['TotalBsmtSF']>0,'HasBsmt'] = 1
    
    print("----------------------------------------------------------------------------")

    

    

## Cited From: https://pythonprogramming.net/working-with-non-numerical-data-machine-learning-tutorial/ 
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

def standardize(df, column):
    df.loc[:,column] = (df.loc[:,column] - df.loc[:,column].mean())/df.loc[:,column].std()
    return df




# ============================================================================

def transformData(trainDF, testDF):
    
  
    
    preprocessing(trainDF, testDF)
    
    standardizeCols = ['OverallQual','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','2ndFlrSF','1stFlrSF','GrLivArea','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','LotFrontage']
    standardize(trainDF, standardizeCols)
    standardize(testDF, standardizeCols)
    
    
    
    predictors = getNumericAttrs(trainDF).drop(['SalePrice','Id','GarageQual','Condition2','BsmtFinType2','BsmtFinSF2','Utilities','Heating','BsmtHalfBath','Functional','GarageCond'])

   
    '''
    You'll want to use far more predictors than just these two columns, of course. But when you add
    more, you'll need to do things like handle missing values and convert non-numeric to numeric.
    Other preprocessing steps would likely be wise too, like standardization, get_dummies, 
    or converting or creating attributes based on your intuition about what's relevant in housing prices.
    '''
    
    trainInput = trainDF.loc[:, predictors]
    testInput = testDF.loc[:, predictors]
    '''
    Any transformations you do on the trainInput will need to be done on the
    testInput the same way. (For example, using the exact same min and max, if
    you're doing normalization.)
    '''
    
    trainOutput = trainDF.loc[:, 'SalePrice']
    testIDs = testDF.loc[:, 'Id']
    
    return trainInput, testInput, trainOutput, testIDs, predictors
    
# ===============================================================================
'''
Demonstrates some provided helper functions that you might find useful.
'''
def demonstrateHelpers(trainDF):
    print("Attributes with missing values:", getAttrsWithMissingValues(trainDF), sep='\n')
    
    numericAttrs = getNumericAttrs(trainDF)
    print("Numeric attributes:", numericAttrs, sep='\n')
    
    nonnumericAttrs = getNonNumericAttrs(trainDF)
    print("Non-numeric attributes:", nonnumericAttrs, sep='\n')

    print("Values, for each non-numeric attribute:", getAttrToValuesDictionary(trainDF.loc[:, nonnumericAttrs]), sep='\n')

# ===============================================================================
'''
Returns a dictionary mapping an attribute to the array of values for that attribute.
'''
def getAttrToValuesDictionary(df):
    attrToValues = {}
    for attr in df.columns.values:
        attrToValues[attr] = df.loc[:, attr].unique()

    return attrToValues

# ===============================================================================
'''
Returns the attributes with missing values.
'''
def getAttrsWithMissingValues(df):
    valueCountSeries = df.count(axis=0)  # 0 to count down the rows
    numCases = df.shape[0]  # Number of examples - number of rows in the data frame
    missingSeries = (numCases - valueCountSeries)  # A Series showing the number of missing values, for each attribute
    attrsWithMissingValues = missingSeries[missingSeries != 0].index
    return attrsWithMissingValues

# =============================================================================

'''
Returns the numeric attributes.
'''
def getNumericAttrs(df):
    return __getNumericHelper(df, True)

'''
Returns the non-numeric attributes.
'''
def getNonNumericAttrs(df):
    return __getNumericHelper(df, False)

def __getNumericHelper(df, findNumeric):
    isNumeric = df.applymap(np.isreal) # np.isreal is a function that takes a value and returns True (the value is real) or False
                                       # applymap applies the given function to the whole data frame
                                       # So this returns a DataFrame of True/False values indicating for each value in the original DataFrame whether it is real (numeric) or not

    isNumeric = isNumeric.all() # all: For each column, returns whether all elements are True
    attrs = isNumeric.loc[isNumeric==findNumeric].index # selects the values in isNumeric that are <findNumeric> (True or False)
    return attrs

# =============================================================================

if __name__ == "__main__":
    main()
    


