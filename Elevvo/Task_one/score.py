import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#Load and Inspect Data
df = pd.read_csv("student_performance.csv")  # dataset from Kaggle
print(df.head())
print(df.info())
print(df.describe())


#Data Cleaning & Visualization
df = df.dropna()
sns.pairplot(df, x_vars=['Hours_Studied'], y_vars='Exam_Score', height=5, aspect=1, kind='reg')
plt.show()


#Split Data
X = df[['Hours_Studied']]   # independent variable
y = df['Exam_Score']      # target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#Train Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


#Evaluate Performance
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# show data
plt.scatter(X_test, y_test, color='blue', label="Actual")
plt.plot(X_test, y_pred, color='red', linewidth=2, label="Predicted")
plt.legend()
plt.xlabel("Hours_Studied")
plt.ylabel("Exam_Score")
plt.title("Linear Regression: Study Hours vs Exam Score")
plt.show()


