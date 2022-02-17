import os
import numpy as np
import pandas as pd
import seaborn as sns


#Increase the printoutput
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width',1000)

#set working directory
os.chdir("C:/Users/DELL/Desktop/Directory")

#read data
rawdf=pd.read_csv("PropertyPrice_Data.csv")
predictiondf=pd.read_csv("PropertyPrice_Prediction.csv")

#Splitting the data into train and test
from sklearn.model_selection import train_test_split
traindf, testdf = train_test_split(rawdf, train_size = 0.8, random_state = 7)

#creae source column in all data sets
traindf['Source'] = "Train"
testdf['Source'] = "Test"
predictiondf['Source'] = "Predcition"

#combine all three dfs
fullraw = pd.concat([traindf, testdf, predictiondf], axis = 0)
fullraw.shape
fullraw.columns
fullraw = fullraw.drop(['Id'], axis = 1)
fullraw.isnull().sum()

fullraw_columns = fullraw.columns

for i in (fullraw_columns):
    
    if (i in ["Sale_Price", "Source"]):
        continue
    
    if traindf[i].dtype == object:
        print("Cat: ", i)
        tempMode = fullraw.loc[fullraw["Source"] == "Train", i].mode()[0]
        fullraw[i].fillna(tempMode, inplace = True)
    else:
        print("Cont: ", i)
        tempMedian = traindf[i].median()
        fullraw[i] = fullraw[i].fillna(tempMedian)


corrDf = fullraw[fullraw["Source"] == "Train"].corr()
# corrDf.head()
sns.heatmap(corrDf, xticklabels=corrDf.columns, yticklabels=corrDf.columns, cmap='YlOrBr')

categoricalVars = traindf.columns[traindf.dtypes == object]
categoricalVars

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt
fileName = "C:/Users/DELL/Desktop/Directory/Categorical_Variables_Analysis.pdf"
pdf = PdfPages(fileName)
for colNumber, colName in enumerate(categoricalVars):
    fig = plt.figure()
    sns.boxplot(y = traindf["Sale_Price"], x = traindf[colName])
    pdf.savefig(colNumber+1)
    
pdf.close()

fullraw2 = pd.get_dummies(fullraw, drop_first = True)

fullraw2.shape
fullraw.shape

from statsmodels.api import add_constant
fullraw2 = add_constant(fullraw2)

fullraw2.shape

traindf = fullraw2[fullraw2['Source_Train'] == 1].drop(['Source_Train', 'Source_Test'], axis = 1).copy()

testdf = fullraw2[fullraw2['Source_Test'] == 1].drop(['Source_Train', 'Source_Test'], axis = 1).copy()

predictiondf = fullraw2[(fullraw2['Source_Train'] == 0) & (fullraw2['Source_Test'] == 0)].drop(['Source_Train', 'Source_Test'], axis = 1).copy()

traindf.shape
testdf.shape
predictiondf.shape

trainx = traindf.drop(['Sale_Price'] , axis = 1).copy()
trainy = traindf['Sale_Price'].copy()
testx = traindf.drop(['Sale_Price'], axis = 1).copy()
testy = testdf['Sale_Price'].copy()


from statsmodels.stats.outliers_influence import variance_inflation_factor

tempMaxVIF = 5 # The VIF that will be calculated at EVERY iteration in while loop
maxVIFCutoff = 5 # 5 is recommended cutoff value for linear regression
trainXCopy = trainx.copy()
counter = 1
highVIFColumnNames = []

while (tempMaxVIF >= maxVIFCutoff):
    
    print(counter)
    
    # Create an empty temporary df to store VIF values
    tempVIFDf = pd.DataFrame()
    
    # Calculate VIF using list comprehension
    tempVIFDf['VIF'] = [variance_inflation_factor(trainXCopy.values, i) for i in range(trainXCopy.shape[1])]
    
    # Create a new column "Column_Name" to store the col names against the VIF values from list comprehension
    tempVIFDf['Column_Name'] = trainXCopy.columns
    
    # Drop NA rows from the df - If there is some calculation error resulting in NAs
    tempVIFDf.dropna(inplace=True)
    
    # Sort the df based on VIF values, then pick the top most column name (which has the highest VIF)
    tempColumnName = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,1]
    # tempColumnName = tempVIFDf.sort_values(["VIF"], ascending = True)[-1:]["Column_Name"].values[0]
    
    # Store the max VIF value in tempMaxVIF
    tempMaxVIF = tempVIFDf.sort_values(["VIF"], ascending = False).iloc[0,0]
    # tempMaxVIF = tempVIFDf.sort_values(["VIF"])[-1:]["VIF"].values[0]
    
    print(tempColumnName)
    
    if (tempMaxVIF >= maxVIFCutoff): # This condition will ensure that columns having VIF lower than 5 are NOT dropped
        
        # Remove the highest VIF valued "Column" from trainXCopy. As the loop continues this step will keep removing highest VIF columns one by one 
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highVIFColumnNames.append(tempColumnName)
    
    counter = counter + 1

highVIFColumnNames

highVIFColumnNames.remove('const') # We need to exclude 'const' column from getting dropped/ removed. This is intercept.
highVIFColumnNames

trainX = trainx.drop(highVIFColumnNames, axis = 1)
testX = testx.drop(highVIFColumnNames, axis = 1)
predictiondf = predictiondf.drop(highVIFColumnNames, axis = 1)

trainX.shape
testX.shape
predictiondf.shape

from statsmodels.api import OLS
m1ModelDef = OLS(trainy, trainx) # (Dep_Var, Indep_Vars) # This is model definition
m1ModelBuild = m1ModelDef.fit() # This is model building
m1ModelBuild.summary() # This is model output summary

dir(m1ModelBuild)
m1ModelBuild.pvalues


tempMaxPValue = 0.1
maxPValueCutoff = 0.1
trainXCopy = trainX.copy()
counter = 1
highPValueColumnNames = []


while (tempMaxPValue >= maxPValueCutoff):
    
    print(counter)    
    
    tempModelDf = pd.DataFrame()    
    Model = OLS(trainy, trainXCopy).fit()
    tempModelDf['PValue'] = Model.pvalues
    tempModelDf['Column_Name'] = trainXCopy.columns
    tempModelDf.dropna(inplace=True) # If there is some calculation error resulting in NAs
    tempColumnName = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,1]
    tempMaxPValue = tempModelDf.sort_values(["PValue"], ascending = False).iloc[0,0]
    
    if (tempMaxPValue >= maxPValueCutoff): # This condition will ensure that ONLY columns having p-value lower than 0.1 are NOT dropped
        print(tempColumnName, tempMaxPValue)    
        trainXCopy = trainXCopy.drop(tempColumnName, axis = 1)    
        highPValueColumnNames.append(tempColumnName)
    
    counter = counter + 1
    
highPValueColumnNames

Model.summary()

trainX = trainx.drop(highPValueColumnNames, axis = 1)
testX = testxdrop(highPValueColumnNames, axis = 1)
predictiondf = predictiondf.drop(highPValueColumnNames, axis = 1)

trainX.shape
testX.shape

Model = OLS(trainy, trainX).fit()
Model.summary()

Test_Pred = Model.predict(testX)
Test_Pred[0:6]
testy[:6]

import seaborn as sns

sns.scatterplot(Model.fittedvalues, Model.resid)

# Normality of errors check
sns.distplot(Model.resid) # Should be somewhat close to normal distribution

np.sqrt(np.mean((testy - Test_Pred)**2))