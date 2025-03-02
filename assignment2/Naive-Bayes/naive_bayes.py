# -------------------------------------------------------------------------
# AUTHOR: Brandon Tang
# FILENAME: naive_bayes.py
# SPECIFICATION: Reads the file weather_training.csv (training set) and output the classification of each of the 10 instances from the file weather_test (test set) if the classification confidence is >= 0.75.
#                Sample of output:
#                Day    Outlook  Temperature Humidity Wind  PlayTennis Confidence
#                D1003  Sunny    Cool        High     Weak  No         0.86
#                D1005  Overcast Mild        High     Weak  Yes        0.78
# FOR: CS 4210- Assignment #2
# TIME SPENT: 50 minutes
# -----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

# Importing some Python libraries
from sklearn.naive_bayes import GaussianNB


# Function to read a CSV file and return the data as a list of dictionaries
def read_csv(filename):
    data = []
    with open(filename, "r") as file:
        lines = file.readlines()
        headers = lines[0].strip().split(",")
        for line in lines[1:]:
            values = line.strip().split(",")
            row = {headers[i]: values[i] for i in range(len(headers))}
            data.append(row)
    return data


# Function to transform categorical features into numerical values
def transform_features(data, feature_mappings):
    transformed_data = []
    for row in data:
        transformed_row = []
        for feature, mapping in feature_mappings.items():
            transformed_row.append(mapping[row[feature]])
        transformed_data.append(transformed_row)
    return transformed_data


# Function to transform classes into numerical values
def transform_classes(data, class_mapping, target_column):
    transformed_classes = []
    for row in data:
        transformed_classes.append(class_mapping[row[target_column]])
    return transformed_classes


# Define mappings for categorical features and classes
feature_mappings = {
    "Outlook": {"Sunny": 1, "Overcast": 2, "Rain": 3},
    "Temperature": {"Cool": 1, "Mild": 2, "Hot": 3},
    "Humidity": {"Normal": 1, "High": 2},
    "Wind": {"Weak": 1, "Strong": 2},
}
class_mapping = {"Yes": 1, "No": 2}

# Reading the training data in a csv file
training_data = read_csv("weather_training.csv")

# Transform the original training features to numbers and add them to the 4D array X.
# For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
X = transform_features(training_data, feature_mappings)

# Transform the original training classes to numbers and add them to the vector Y.
# For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
Y = transform_classes(training_data, class_mapping, "PlayTennis")

# Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

# Reading the test data in a csv file
test_data = read_csv("weather_test.csv")

# Transform the test features to numbers
X_test = transform_features(test_data, feature_mappings)

# Printing the header of the solution
print(
    "{:<8} {:<10} {:<12} {:<10} {:<8} {:<12} {:<10}".format(
        "Day", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis", "Confidence"
    )
)

# Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for i, row in enumerate(test_data):
    probabilities = clf.predict_proba([X_test[i]])[0]
    confidence = max(probabilities)
    predicted_class = clf.predict([X_test[i]])[0]
    if confidence >= 0.75:
        predicted_label = list(class_mapping.keys())[
            list(class_mapping.values()).index(predicted_class)
        ]
        print(
            "{:<8} {:<10} {:<12} {:<10} {:<8} {:<12} {:<10.2f}".format(
                row["Day"],
                row["Outlook"],
                row["Temperature"],
                row["Humidity"],
                row["Wind"],
                predicted_label,
                confidence,
            )
        )
