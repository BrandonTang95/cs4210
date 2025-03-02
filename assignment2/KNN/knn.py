# -------------------------------------------------------------------------
# AUTHOR: Brandon Tang
# FILENAME: knn.py
# SPECIFICATION: Reads the file email_classification.csv and compute the LOO-CV error rate for a 1NN classifier on the spam/ham classification task.
#                The dataset consists of email samples, where each sample includes the counts of 20 specific words (e.g., “agenda” or “prize”) representing their frequency of occurrence.
#
# FOR: CS 4210- Assignment #2
# TIME SPENT: 45 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

# Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

# Initialize variables
db = []
X = []
Y = []

# Reading the data in a csv file
with open("email_classification.csv", "r") as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:  # skipping the header
            db.append(row)

# Transform the original categorical training features to numbers and add to the 20D array X.
# For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
# Convert each feature value to float to avoid warning messages
for row in db:
    features = [float(value) for value in row[:-1]]  # Exclude the last column (label)
    X.append(features)

# Transform the original categorical training classes to numbers and add to the vector Y.
# For instance, Y = [1, 2, ,...].
# Convert each feature value to float to avoid warning messages
for row in db:
    label = 1 if row[-1] == "spam" else 0  # Convert "spam" to 1 and "ham" to 0
    Y.append(label)

# Initialize variables for LOO-CV
total_misclassifications = 0
total_predictions = len(db)

# Loop your data to allow each instance to be your test set
for i in range(len(db)):
    # Split the data into training and test sets
    X_train = X[:i] + X[i + 1 :]  # Exclude the current test point
    Y_train = Y[:i] + Y[i + 1 :]  # Exclude the current test point
    X_test = [X[i]]  # Current test point
    Y_test = Y[i]  # True label of the current test point

    # Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)  # p=2 for Euclidean distance
    clf.fit(X_train, Y_train)

    # Use your test sample in this iteration to make the class prediction
    class_predicted = clf.predict(X_test)[0]

    # Compare the prediction with the true label of the test instance to start calculating the error rate
    if class_predicted != Y_test:
        total_misclassifications += 1

# Calculate the error rate
error_rate = total_misclassifications / total_predictions

# Print the error rate
print(f"LOO-CV Error Rate for 1NN: {error_rate}")
