
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_rows', None)


df=pd.read_csv('C:\\Users\\naval\\Downloads\\q\\f\\House_Price.csv')

df.drop(columns=['PoolQC','MiscFeature','Fence','Alley'],inplace=True)
df['LotFrontage'].fillna(df['LotFrontage'].mean(),inplace=True)
df['MasVnrType'].fillna(0,inplace=True)
df['MasVnrArea'].fillna(0,inplace=True)
df['BsmtQual'].fillna(method='bfill',inplace=True)
df['BsmtCond'].fillna(method='bfill',inplace=True)
df['BsmtFinType1'].fillna(method='bfill',inplace=True)
df['BsmtFinType2'].fillna(method='bfill',inplace=True)
df['BsmtExposure'].fillna('No',inplace=True)
df['Electrical'].fillna('SBrkr',inplace=True)
df['FireplaceQu'].fillna(method='bfill',inplace=True)
df['GarageType'].fillna(method='bfill',inplace=True)
df['GarageYrBlt'].fillna(method='bfill',inplace=True)
df['GarageFinish'].fillna(method='bfill',inplace=True)
df['GarageQual'].fillna(method='bfill',inplace=True)
df['GarageCond'].fillna(method='bfill',inplace=True)
df['FireplaceQu'].fillna(method='ffill',inplace=True)



from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
lb.fit(df['Neighborhood'])
df['Neighborhoo']=lb.transform(df['Neighborhood'])

lb.fit(df['Condition1'])
df['Condition1']=lb.transform(df['Condition1'])

lb.fit(df['Condition2'])
df['Condition2']=lb.transform(df['Condition2'])

lb.fit(df['BldgType'])
df['BldgType']=lb.transform(df['BldgType'])

lb.fit(df['HouseStyle'])
df['HouseStyle']=lb.transform(df['HouseStyle'])

lb.fit(df['RoofStyle'])
df['RoofStyle']=lb.transform(df['RoofStyle'])

lb.fit(df['RoofMatl'])
df['RoofMatl']=lb.transform(df['RoofMatl'])


lb.fit(df['Exterior1st'])
df['Exterior1st']=lb.transform(df['Exterior1st'])

lb.fit(df['Exterior2nd'])
df['Exterior2nd']=lb.transform(df['Exterior2nd'])


lb.fit(df['ExterQual'])
df['ExterQual']=lb.transform(df['ExterQual'])


lb.fit(df['ExterCond'])
df['ExterCond']=lb.transform(df['ExterCond'])

lb.fit(df['Foundation'])
df['Foundation']=lb.transform(df['Foundation'])




lb.fit(df['BsmtQual'])
df['BsmtQual']=lb.transform(df['BsmtQual'])

lb.fit(df['BsmtCond'])
df['BsmtCond']=lb.transform(df['BsmtCond'])

lb.fit(df['BsmtExposure'])
df['BsmtExposure']=lb.transform(df['BsmtExposure'])

lb.fit(df['BsmtFinType1'])
df['BsmtFinType1']=lb.transform(df['BsmtFinType1'])

lb.fit(df['BsmtFinType2'])
df['BsmtFinType2']=lb.transform(df['BsmtFinType2'])

lb.fit(df['Heating'])
df['Heating']=lb.transform(df['Heating'])

lb.fit(df['HeatingQC'])
df['HeatingQC']=lb.transform(df['HeatingQC'])

lb.fit(df['CentralAir'])
df['CentralAir']=lb.transform(df['CentralAir'])


lb.fit(df['Electrical'])
df['Electrical']=lb.transform(df['Electrical'])

lb.fit(df['KitchenQual'])
df['KitchenQual']=lb.transform(df['KitchenQual'])

lb.fit(df['Functional'])
df['Functional']=lb.transform(df['Functional'])


lb.fit(df['FireplaceQu'])
df['FireplaceQu']=lb.transform(df['FireplaceQu'])

lb.fit(df['GarageType'])
df['GarageType']=lb.transform(df['GarageType'])

lb.fit(df['GarageFinish'])
df['GarageFinish']=lb.transform(df['GarageFinish'])

lb.fit(df['GarageQual'])
df['GarageQual']=lb.transform(df['GarageQual'])

lb.fit(df['GarageCond'])
df['GarageCond']=lb.transform(df['GarageCond'])


lb.fit(df['PavedDrive'])
df['PavedDrive']=lb.transform(df['PavedDrive'])


lb.fit(df['SaleType'])
df['SaleType']=lb.transform(df['SaleType'])

lb.fit(df['SaleCondition'])
df['SaleCondition']=lb.transform(df['SaleCondition'])

lb.fit(df['Neighborhood'])
df['Neighborhood']=lb.transform(df['Neighborhood'])

df.replace({'MSZoning':{'RL':0,'RM':1,'FV':2,'RH':3,'C (all)':4},'Street':{'Pave':1,'Grvl':0},'LotShape':{'Reg':0,'IR1':1,'IR2':2,'IR3':3}},inplace=True)
df.replace({'LandContour':{'Lvl':0,'Bnk':1,'HLS':2,'Low':3},'Utilities':{'AllPub':0,'NoSeWa':1},'LotConfig':{'Inside':0,'Corner':1,'CulDSac':2,'FR2':3,'FR3':4}},inplace=True)
df.replace({'LandSlope':{'Gtl':0,'Mod':1,'Sev':2},'MasVnrType':{'BrkFace':1,'Stone':2,'BrkCmn':3}},inplace=True)

#plt.figure(figsize=(9, 8))
#sns.distplot(df['SalePrice'], color='g', bins=100, hist_kws={'alpha': 0.4})

#df.hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
#plt.show()

x=df.iloc[:,0:-1]
y=df.iloc[:,-1]


X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=.2,random_state=2)



from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(X_train,Y_train)

y_pred=lr.predict(X_test)
prediction=r2_score(y_pred,Y_test)
print(prediction)

