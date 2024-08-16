import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
pd.set_option('display.max_rows', None)
# Load the data
df = pd.read_csv('C:\\Users\\naval\\Downloads\\q\\f\\House_Price.csv')

# Check for required columns
required_columns = ['Id', 'SalePrice']
if not all(col in df.columns for col in required_columns):
    raise ValueError(f"Missing required columns: {', '.join(set(required_columns) - set(df.columns))}")

# Drop irrelevant columns
df.drop(columns=['PoolQC', 'MiscFeature', 'Fence', 'Alley'], inplace=True)

# Handle missing values
df['LotFrontage'].fillna(df['LotFrontage'].mean(), inplace=True)
df['MasVnrType'].fillna('None', inplace=True)
df['MasVnrArea'].fillna(0, inplace=True)
df['BsmtQual'].fillna('No', inplace=True)
df['BsmtCond'].fillna('No', inplace=True)
df['BsmtExposure'].fillna('No', inplace=True)
df['BsmtFinType1'].fillna('No', inplace=True)
df['BsmtFinType2'].fillna('No', inplace=True)
df['Electrical'].fillna('SBrkr', inplace=True)
df['FireplaceQu'].fillna('No', inplace=True)
df['GarageType'].fillna('No', inplace=True)
df['GarageFinish'].fillna('No', inplace=True)
df['GarageQual'].fillna('No', inplace=True)
df['GarageCond'].fillna('No', inplace=True)

# Encode categorical features
categorical_features = ['Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
                        'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
                        'CentralAir', 'Electrical','GarageYrBlt', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
                        'SaleType', 'SaleCondition']

lb = LabelEncoder()
for column in categorical_features:
    if column in df.columns:
        df[column] = lb.fit_transform(df[column].astype(str))

# Replace specific values
df.replace({'MSZoning': {'RL': 0, 'RM': 1, 'FV': 2, 'RH': 3, 'C (all)': 4},
            'Street': {'Pave': 1, 'Grvl': 0},
            'LotShape': {'Reg': 0, 'IR1': 1, 'IR2': 2, 'IR3': 3},
            'LandContour': {'Lvl': 0, 'Bnk': 1, 'HLS': 2, 'Low': 3},
            'Utilities': {'AllPub': 0, 'NoSeWa': 1},
            'LotConfig': {'Inside': 0, 'Corner': 1, 'CulDSac': 2, 'FR2': 3, 'FR3': 4},
            'LandSlope': {'Gtl': 0, 'Mod': 1, 'Sev': 2},
            'MasVnrType': {'BrkFace': 1, 'Stone': 2, 'BrkCmn': 3, 'None': 0}}, inplace=True)

# Ensure no missing values
print(df.isnull().sum())
if df.isnull().sum().sum() > 0:
    raise ValueError("There are still missing values in the dataset.")

# Prepare the data for training
X = df.drop(['Id', 'SalePrice'], axis=1)
y = df['SalePrice']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Prepare the test data for prediction
# In a real scenario, you should have a separate test dataset
test_data = X  # Using the same data for simplicity, but in practice, use separate test data

# Make predictions
preds = model.predict(test_data)

# Create the output DataFrame
ids = df['Id']
output = pd.DataFrame({'Id': ids, 'SalePrice': preds})

# Display the first few rows of the output
def predict_price_by_id(input_id):
    """
    Predict the SalePrice for a given Id.
    
    Parameters:
    - input_id (int): The Id for which to predict the SalePrice.
    
    Returns:
    - The predicted SalePrice.
    """
    # Ensure the Id exists in the dataset
    if input_id not in df['Id'].values:
        return "Id not found in the dataset."
    
    # Locate the row with the given Id
    row = df[df['Id'] == input_id]
    
    # Prepare the data for prediction
    row_data = row.drop(['Id', 'SalePrice'], axis=1)
    
    # Predict the SalePrice
    predicted_price = model.predict(row_data)
    
    return predicted_price[0]

# Example usage
input_id = 12345  # Replace with the desired Id
predicted_price = predict_price_by_id(1200)
print(f"The predicted SalePrice for Id {input_id} is: {predicted_price}")