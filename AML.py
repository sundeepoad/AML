import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("LI-Small_Trans.csv")

print(df.head(10))

# Check for missing values
print(df.isna().sum())

print(df.info())


## Converting timestamp column to date time.
df['Timestamp'] = pd.to_datetime(df['Timestamp'])


## Plot to show what payment type distribution
##sns.countplot(x = 'Payment Format', data = df, palette = 'cool')
##plt.title("Payment Fromats")
##plt.xlabel('Payment Format')
##plt.ylabel('Count of Each payment')
##plt.show()



count_laundering = df['Is Laundering'].value_counts()
print(count_laundering)

## Total non-Luandering records: 6920484
## Laundering records: 3565

## The data is very imbalanced. Need to down sample since Dataset is large, downsampling Non-Laundering cases makes more sense.

laundering = df[df['Is Laundering'] == 1]
non_laundering = df[df['Is Laundering'] == 0]

down_sample = non_laundering.sample(n = len(laundering), random_state = 42)

balanced = pd.concat([down_sample, laundering])


count = balanced['Is Laundering'].value_counts()
print(count, "balanced")

## After balancing:  3565 cases for both laundering and non_laundering trasactions.


lf = balanced[balanced["Is Laundering"] == 1]
mean_amount = lf["Amount Paid"].mean()
mode_amount = lf["Amount Paid"].mode()
print("Mean:", mean_amount)
print("Mode:", mode_amount)


## Encoding labels

balanced['Timestamp'] = pd.to_datetime(balanced['Timestamp'])
balanced['Year'] = balanced['Timestamp'].dt.year
balanced['Month'] = balanced['Timestamp'].dt.month
balanced['Day'] = balanced['Timestamp'].dt.day
balanced['Hour'] = balanced['Timestamp'].dt.hour
balanced = balanced.drop(columns=['Timestamp'])


le = LabelEncoder()

balanced['Payment Currency'] = le.fit_transform(balanced['Payment Currency'])
balanced['Receiving Currency'] = le.fit_transform(balanced['Receiving Currency'])
balanced['Payment Format'] = le.fit_transform(balanced['Payment Format'])

print("After Label encoding: ", balanced.head(5))


X = balanced.drop(columns = ['Is Laundering', 'Account', 'Account1'])
y = balanced['Is Laundering']


## Splitting data to test and train
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 43)

xg = xgb.XGBClassifier(n_estimators = 100, max_depth = 6, learning_rate = 0.1)
xg.fit(X_train, y_train)

y_pred = xg.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy: ", acc)



print(classification_report(y_test, y_pred))


important_features = xg.feature_importances_

plt.barh(range(len(important_features)), important_features, align = 'center')
plt.yticks(range(len(important_features)), X_train.columns)
plt.xlabel("Feature Importance")
plt.title("Important Features")
plt.show()


## As per the plot, Payment format is most important feature, From Bank is second most important.

