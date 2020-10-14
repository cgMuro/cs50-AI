import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    # Get data from file
    with open(filename) as f:
        csv_reader = csv.reader(f)
        next(csv_reader)

        data = []
        for row in csv_reader:
            data.append({
                'evidence': [cell for cell in row[:17]],
                'label': 0 if row[17] == "FALSE" else 1
            })

        # Transfor all the data to numbers
        months = {
            'Jan': 0,
            'Feb': 1,
            'Mar': 2, 
            'Apr': 3,
            'May': 4,
            'June': 5, 
            'Jul': 6,
            'Aug': 7,
            'Sep': 8,
            'Oct': 9,
            'Nov': 10,
            'Dec': 11
        }
        for i in data:
            j = i['evidence']
            j[0] = int(j[0])
            j[1] = float(j[1])
            j[2] = int(j[2])
            j[3] = float(j[3])
            j[4] = int(j[4])
            j[5] = float(j[5])
            j[6] = float(j[6])
            j[7] = float(j[7])
            j[8] = float(j[8])
            j[9] = float(j[9])
            j[10] = months[j[10]]
            j[11] = int(j[11])
            j[12] = int(j[12])
            j[13] = int(j[13])
            j[14] = int(j[14])
            j[15] = 1 if j[15] == 'Returning_Visitor' else 0
            j[16] = 0 if j[16] == 'FALSE' else 1
        
        return (
            [i['evidence'] for i in data],
            [i['label'] for i in data]
        )

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # Create model
    model = KNeighborsClassifier(n_neighbors=1)
    X_training = evidence
    y_training = labels
    # Train model
    model.fit(X_training, y_training)
    return model

def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    # Calculate sensitivity and specificity
    positive = 0
    sensitivity = 0
    negative = 0
    specificity = 0

    for labels, predictions in zip(labels, predictions):
        if labels == 1:
            positive += 1
            if labels == predictions:
                sensitivity += 1

        if labels == 0:
            negative += 1
            if labels == predictions:
                specificity += 1
        
    sensitivity /= positive
    specificity /= negative

    return sensitivity, specificity


if __name__ == "__main__":
    main()
