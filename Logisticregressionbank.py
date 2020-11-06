import pandas as pd
import matplotlib.pyplot as plt
import pip
from sklearn.linear_model import LogisticRegression

dataframe= pd.read_csv("bank-full.csv")
print(dataframe.head)
y= dataframe.iloc[:,16]
dataframe=pd.get_dummies(dataframe,drop_first=True,columns=['job','marital','education','default','housing','loan','contact','month','poutcome','y'])

print(dataframe.head)

y= dataframe.iloc[:,16]
print(y.shape)
X= dataframe.iloc[:,0:16]

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
print(len(os_data_X))
print(len(os_data_y))

os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
print(len(os_data_X))
print(len(os_data_y))
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

X=os_data_X
y=os_data_y
model=LogisticRegression(max_iter=1000)
model.fit(X,y)
y_pred = model.predict(X_test)


model.coef_ # coefficients of features

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

confusion_matrix = confusion_matrix(y_test,y_pred)
print(confusion_matrix)
classfication_matrix= classification_report(y_test,y_pred)
print(classfication_matrix)

