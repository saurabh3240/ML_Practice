# Import the Pandas library
import  pandas as pd
#import numpy
import numpy as np

#import scikit-learn library
from sklearn import tree

# Load the train and test datasets to create two DataFrames
train_url = "DATA/train.csv"
train = pd.read_csv(train_url)
test_url = "DATA/test.csv"
test = pd.read_csv(test_url)

#Print the `head` of the train and test dataframes
#print(train.head())  first 5 rows of data 

# describe return max,min std,etc
#print(train.describe())
#print(train.shape)  # rows and column

# Passengers that survived vs passengers that passed away
#print(train["Survived"].value_counts())

# As proportions
#print(train["Survived"].value_counts(normalize=True))

# Males that survived vs males that passed away
#print(train["Survived"][train["Sex"]=='male'].value_counts())

# Females that survived vs Females that passed away
#print(train["Survived"][train["Sex"]=='female'].value_counts())

# Normalized male survival
#print(train["Survived"][train["Sex"]=='male'].value_counts(normalize = True))

# Normalized female survival
#print(train["Survived"][train["Sex"]=='female'].value_counts(normalize = True))


# Create the column Child and assign to 'NaN'
#train["Child"] = float('NaN')

# Assign 1 to passengers under 18, 0 to those 18 or older. Print the new column.
#train["Child"][train["Age"]<18] =1
#train["Child"][train["Age"]>=18] =0
#print(train["Child"])


# Print normalized Survival Rates for passengers under 18
#print(train["Survived"][train["Child"] == 1].value_counts(normalize = True))

# Print normalized Survival Rates for passengers 18 or older
#print(train["Survived"][train["Child"] == 0].value_counts(normalize = True))



# Create a copy of test: test_one
test_one = test;

# Initialize a Survived column to 0
test_one["Survived"]=0;

# Set Survived to 1 if Sex equals "female" and print the `Survived` column from `test_one`
#test_one["Survived"][test_one["Sex"]=='female']  = 1;
#print(test_one["Survived"])


############implement Decision Tree ########

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna('S')

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
train["Age"].fillna( train["Age"].median(),inplace = True);
# Create the target and features numpy arrays: target, features_one
target = train["Survived"].values  # target feature  = Y_train
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one,target)

# Look at the importance and score of the included features
#print(my_tree_one.feature_importances_) 
#print(my_tree_one.score(features_one,target))

#print(test)
# Impute the missing value with the median
test.Fare[152] = test.Fare.median();

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test["Age"].fillna( test["Age"].median(),inplace = True);


# Extract the features from the test set: Pclass, Sex, Age, and Fare.
test_features = test[['Pclass', 'Sex', 'Age', 'Fare']].values

# Make your prediction using the test set
my_prediction = my_tree_one.predict(test_features)
#print(my_prediction)
x = test["PassengerId"]
print(x)
# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
print(PassengerId)
my_solution = pd.DataFrame(my_prediction ,PassengerId,columns =["Survived"])
print("Saurabh")
print(my_solution)


# write to upload output file
csv_file = [my_solution["Survived"]];
csv_file.to_csv("a.csv",index=False,);