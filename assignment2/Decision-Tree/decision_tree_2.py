# -------------------------------------------------------------------------
# AUTHOR: Brandon Tang
# FILENAME: decision_tree_2.py
# SPECIFICATION: Reads the files contact_lens_training_1.csv, contact_lens_training_2.csv, and contact_lens_training_3.csv.
#                Each training set has a different number of instances (10, 100, 1000 samples).
#                You will observe that the trees are being created by setting the parameter max_depth = 5, which is used to define the maximum depth of the tree (pre-pruning strategy) in sklearn.
#                The goal is to train, test, and output the performance of the 3 models created by using each training set on the test set provided (contact_lens_test.csv).
#                This process is repeated 10 times (train and test using a different training set), choosing the average accuracy as the final classification performance of each model.

# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# Importing some Python libraries
from sklearn import tree
import csv

dataSets = [
    "contact_lens_training_1.csv",
    "contact_lens_training_2.csv",
    "contact_lens_training_3.csv",
]

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    # Reading the training data in a csv file
    with open(ds, "r") as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0:  # skipping the header
                dbTraining.append(row)

    # Define mappings for categorical features and labels
    age_mapping = {"Young": 1, "Prepresbyopic": 2, "Presbyopic": 3}
    prescription_mapping = {"Myope": 1, "Hypermetrope": 2}
    astigmatism_mapping = {"No": 1, "Yes": 2}
    tear_mapping = {"Normal": 1, "Reduced": 2}
    label_mapping = {"Yes": 1, "No": 2}

    # Transform the original categorical training features to numbers and add to the 4D array X.
    # For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for row in dbTraining:
        X.append(
            [
                age_mapping[row[0]],
                prescription_mapping[row[1]],
                astigmatism_mapping[row[2]],
                tear_mapping[row[3]],
            ]
        )

    # Transform the original categorical training classes to numbers and add to the vector Y.
    # For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    Y = [label_mapping[row[4]] for row in dbTraining]

    # Loop your training and test tasks 10 times here
    total_accuracy = 0  # Reset total_accuracy for each dataset

    for i in range(10):
        # Fitting the decision tree to the data setting max_depth=5
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
        clf = clf.fit(X, Y)

        # Read the test data and add this data to dbTest
        dbTest = []
        with open("contact_lens_test.csv", "r") as csvfile:
            reader = csv.reader(csvfile)
            for i, row in enumerate(reader):
                if i > 0:  # skipping the header
                    dbTest.append(row)

        correct_predictions = 0
        total_predictions = len(dbTest)

        for data in dbTest:
            # Transform the features of the test instances to numbers following the same strategy done during training,
            # and then use the decision tree to make the class prediction.
            test_features = [
                age_mapping[data[0]],
                prescription_mapping[data[1]],
                astigmatism_mapping[data[2]],
                tear_mapping[data[3]],
            ]
            class_predicted = clf.predict([test_features])[0]

            # Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            true_label = label_mapping[data[4]]
            if class_predicted == true_label:
                correct_predictions += 1

        accuracy = correct_predictions / total_predictions
        total_accuracy += accuracy  # Accumulate accuracy for each run

    # Find the average of this model during the 10 runs (training and test set)
    avg_accuracy = total_accuracy / 10

    # Print the average accuracy of this model during the 10 runs (training and test set).
    # Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f"Final accuracy when training on {ds}: {avg_accuracy}")
