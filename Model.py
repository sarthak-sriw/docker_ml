import pandas as pd
import joblib 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Loading the Data as Pandas DataFrame
df = pd.read_csv("Salary_Data.csv")

# Preprocessing the Data according to our needs
X = df["YearsExperience"].values.reshape(30,1)
y = df["Salary"]

# Spliting the Data into Training Set(80%) and Test Set(20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initated the Machine Learning Model
l = LinearRegression()

# Trained the Machine Learning Model using X_train, y_train data 
l.fit(X_train ,y_train)

# Saving the Model as pickle file
joblib.dump(l ,"SalaryModel.pkl")
