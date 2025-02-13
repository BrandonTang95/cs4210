#-------------------------------------------------------------------------
# AUTHOR: Brandon Tang
# FILENAME: decision_tree.py
# SPECIFICATION: This program will read the file contact_lens.csv and output a decision tree of it.
# FOR: CS 4210- Assignment #1
# TIME SPENT: 2 hours
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

# reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)
         print(row)

# transform the original categorical training features into numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
# --> add your Python code here
age = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
spectacle = {"Myope": 1, "Hypermetrope": 2}
astigmatism = {"No": 1, "Yes": 2}
tear = {"Reduced": 1, "Normal": 2}
recommendation = {"Yes": 1, "No": 2}

# transform the original categorical training classes into numbers and add to the vector Y. For instance Yes = 1, No = 2
# --> add your Python code here
for row in db:
    X.append([
        age[row[0]], 
        spectacle[row[1]], 
        astigmatism[row[2]], 
        tear[row[3]]
    ])
    Y.append(recommendation[row[4]])

# fitting the decision tree to the data
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(X, Y)

# plotting the decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()