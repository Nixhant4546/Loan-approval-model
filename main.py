#%% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn import tree

print("hello to my miniproject")
dataset = pd.read_csv("loan-train.csv")
print("This is the dataset head\n",dataset.head()) #print
print("This is the dataset shape\n",dataset.shape) #find rows and column
print("This is the dataset info\n",dataset.info()) # to find missing values and other info
print("This is the dataset infomration like mean etc\n",dataset.describe()) # few information like mean and all
print(pd.crosstab(dataset['Credit_History'],dataset['Loan_Status'], margins=True)) # to find how credit history affects the loan status
dataset.boxplot(column='ApplicantIncome') # visualizing
plt.hist(dataset['ApplicantIncome'], bins=20)
plt.xlabel('ApplicantIncome')
plt.ylabel('Frequency')
plt.title('Histogram of ApplicantIncome')
plt.show()
plt.hist(dataset['CoapplicantIncome'], bins=20)
plt.xlabel('CoapplicantIncome')
plt.ylabel('Frequency')
plt.title('Histogram of CoapplicantIncome')
plt.show()# see if its skewed
dataset.boxplot(column='ApplicantIncome',by ='Education') #shows if education does matter
# see if there are outliers - a thing situated away or detached from the main body or system
plt.boxplot(dataset['LoanAmount'])
plt.ylabel('LoanAmount')
plt.title('Box Plot of LoanAmount')
plt.show()
# see if its skewed if yes then take log
plt.hist(dataset['LoanAmount'], bins=20)
plt.xlabel('LoanAmount')
plt.ylabel('Frequency')
plt.title('Histogram of LoanAmount')
plt.show()
dataset['LoanAmount_log'] = np.log(dataset['LoanAmount'])
plt.hist(dataset['LoanAmount_log'], bins=20)
plt.xlabel('LoanAmount_log')
plt.ylabel('Frequency')
plt.title('Histogram of LoanAmount_log')
plt.show()
# now the data looks normal

#lets find missing values
print("The missing values:\n",dataset.isnull().sum()) # print
#now lets solve those missing values
dataset['Gender'].fillna(dataset['Gender'].mode()[0],inplace=True) #it will find mode of values that is not null and placed where its missing
dataset['Married'].fillna(dataset['Married'].mode()[0],inplace=True)
dataset['Dependents'].fillna(dataset['Dependents'].mode()[0],inplace=True) #mode is only for categorical values
dataset['Self_Employed'].fillna(dataset['Self_Employed'].mode()[0],inplace=True)
dataset.LoanAmount = dataset.LoanAmount.fillna(dataset.LoanAmount.mean())
dataset.LoanAmount_log = dataset.LoanAmount_log.fillna(dataset.LoanAmount_log.mean())
dataset['Loan_Amount_Term'].fillna(dataset['Loan_Amount_Term'].mode()[0],inplace=True)
dataset['Credit_History'].fillna(dataset['Credit_History'].mode()[0],inplace=True)
#now all the missing values are handled
#check if hogaya missing values 
print("After cleaning the missing values:\n",dataset.isnull().sum()) #print

#we know that applicant and coapplicant income is not proper so instead of handling it correctly we'll consider them together
# as total income variable
dataset['TotalIncome'] = dataset['ApplicantIncome'] + dataset['CoapplicantIncome'] # to create a new column
dataset['TotalIncome_log']=np.log(dataset['TotalIncome'])
#visualize using histogram
dataset['TotalIncome_log'].hist(bins=20)
plt.hist(dataset['TotalIncome_log'], bins=20)
plt.xlabel('TotalIncome_log')
plt.ylabel('Frequency')
plt.title('Histogram of TotalIncome_log')
plt.show()
print(dataset.head()) #print kyuki new values add hue so new columns aagaye hai

#bit tricky
x = dataset.iloc[:, np.r_[1:5, 9:11, 13:15]].values
y = dataset.iloc[:,12].values

#print x and y
print("The independent variables are :\n",x)
print("The dependent variables are :\\n",y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0) 
#random state is 0 cause we dont want to change everytime we run the program
print("The x train just to check values :\n",x_train)
from sklearn.preprocessing import LabelEncoder
labelencoder_x = LabelEncoder()
for i in range(0,5):
    x_train[:,i]=labelencoder_x.fit_transform(x_train[:,i])
x_train[:,7]=labelencoder_x.fit_transform(x_train[:,7])
print("Printing x_train again:\n",x_train)
labelencoder_y=LabelEncoder()
y_train = labelencoder_y.fit_transform(y_train)
print("This is the printed y train for the first time to check values:\n",y_train)

for i in range(0,5):
    x_test[:,i]=labelencoder_x.fit_transform(x_test[:,i])
x_test[:,7]=labelencoder_x.fit_transform(x_test[:,7])
print("Now this is xtest which is used to transform character values to numeric:\n",x_test)
y_test = labelencoder_y.fit_transform(y_test)
print("Printing y test...:\n",y_test)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.fit_transform(x_test)
#tricky

from sklearn import metrics
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(x_train,y_train)
y_pred=dt_classifier.predict(x_test)
print("This is printing y pred a variable to predict the x_test\n:",y_pred)
print("The accuracy of decision tree classifier is ",metrics.accuracy_score(y_pred,y_test))

#we are gonna use naive bayes only to predict the data, the accuracy came as 82%
dataset1 = pd.read_csv("loan-test.csv")
dataset1['Gender'].fillna(dataset1['Gender'].mode()[0],inplace=True) #it will find mode of values that is not null and placed where its missing
dataset1['Married'].fillna(dataset1['Married'].mode()[0],inplace=True)
dataset1['Dependents'].fillna(dataset1['Dependents'].mode()[0],inplace=True) #mode is only for categorical values
dataset1['Self_Employed'].fillna(dataset1['Self_Employed'].mode()[0],inplace=True)
dataset1.LoanAmount = dataset1.LoanAmount.fillna(dataset1.LoanAmount.mean())
dataset1['LoanAmount_log'] = np.log(dataset['LoanAmount'])
dataset1.LoanAmount_log = dataset1.LoanAmount_log.fillna(dataset1.LoanAmount_log.mean())
dataset1['Loan_Amount_Term'].fillna(dataset1['Loan_Amount_Term'].mode()[0],inplace=True)
dataset1['Credit_History'].fillna(dataset1['Credit_History'].mode()[0],inplace=True)

print("The info of test to see missing values:",dataset1.isnull().sum())

dataset1['TotalIncome'] = dataset1['ApplicantIncome'] + dataset1['CoapplicantIncome']
dataset1['TotalIncome_log']=np.log(dataset1['TotalIncome'])

test = dataset1.iloc[:, np.r_[1:5, 9:11, 13:15]].values
for i in range(0,5):  
    test[:,i]=labelencoder_x.fit_transform(test[:,i])
test[:,7]=labelencoder_x.fit_transform(test[:,7])
print("The test value\n",test)
test = ss.fit_transform(test)
pred = dt_classifier.predict(test)
print("The prediction in the form of array is ;\n",pred)
dataset1['Loan_Status'] = ['Loan approved' if pred == 1 else 'Loan Rejected' for pred in pred]
print(dataset1.head()) # to check if that new dataset was created
# now we will take this edited csv file in a new csv file

aftermathdata = dataset1.copy()
aftermathdata.to_csv("loan-test1-edited.csv", index=False)
print("Edited CSV file created successfully.")

#Now this is additional so that we will get the idea of how our tree looks like
print("\n\n\nNow lets try making the decision tree itself:\n")
plot_tree(dt_classifier, filled=True, feature_names=['Gender','Married','Dependents','Education','Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','LoanAmount_log','TotalIncome','TotalIncome_log'], class_names=['Loan approved','Loan Rejected'])
plt.show()
'''dot_data = tree.export_graphviz(dt_classifier, out_file=None, feature_names=feature_names, class_names=['No', 'Yes'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('decision_tree')'''


'''
feature_importances = dt_classifier.feature_importances_
#Sort the features by their importance
sorted_features = sorted(feature_importances.item(), key=lambda x: x[1], reverse=True)
#Print the top 5 features and their importances
print("Top 5 Features and Their Importances:")
for feature, importance in sorted_features[:5]:
    print(f"{feature}: {importance:.4f}")
#Visualize the feature importances using a bar chart

plt.bar(range(len(feature_importances)), feature_importances.values())
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.title("Feature Importances")
plt.show()
'''