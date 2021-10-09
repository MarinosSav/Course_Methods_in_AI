import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score

# ML training + baselines
def predict_baseline1(baseline1, chat_input):
    """Makes and returns a prediction of the action label given the user chat input (chat_input) using the dummy
    classifier (baseline1)."""
    return baseline1.predict(chat_input)[0]


RULES = [("thankyou", ["thank you", "thanks"]),
         ("bye", [" bye ", "good bye", "see you", "good night"]),
         ("reqalts", ["how about", "what about", "next", "else", "another", "any other", "anything else"]),
         ("confirm", ["is there", "is it", "does it", "is that", "do they"]),
         ("deny", [" not ", "wrong", "change"]),
         ("hello", [" hi ", " hey ", "hello", "halo"]),
         ("negate", [" no "]),
         #("inform", ["looking", "searching", "trying to find", "cheap", "mediterranean", "asian", "bistro", "any"]),
         ("repeat", ["repeat", "go back", "come again"]),
         ("reqmore", ["more"]),
         ("request", ["what", "where", "when", "could", " can", "address", "postcode", "post code",
                      "phone number", "type of food", "kind of food", "price range", "area"]),
         ("restart", ["star over", "reset", "start again"]),
         ("affirm", ["yes", " ye ", " yea ", "uh huh"]),
         ("ack", ["okay", "thatll do", "alright", " kay ", "good"]),
         ("null", ["noise", "sil", "cough", "unintelligible", "noise", "inaudible"])]
def predict_baseline2(chat_input):
    """Makes and returns a prediction of the action label given the user chat input (chat_input) using a keyword
    matching."""

    for rule in RULES:
        for keyword in rule[1]:
            if keyword in chat_input:
                return rule[0]

    # default return value, for all expressions not found in the rules list.
    # since inform is broad we test for the more specific cases before returning
    return "inform"


def fit_classifier(x_train, y_train, type="dummy"):
    """Function for creating and fitting classifier models. Takes as input the training utterances (x_train) and the
    corresponding training action labels (y_train) as well as the type of classifier to initiate; returns the fitted
    classifier model."""

    if type == "tree":
        classifier = tree.DecisionTreeClassifier()
    elif type == "lr":
        classifier = LogisticRegression()
    else:
        classifier = DummyClassifier(strategy="most_frequent")

    return classifier.fit(x_train, y_train)


def predict(classifier, vectorizer, chat_input):
    """Makes and returns a prediction of the action label given the user chat input (chat_input) and fitted count
     vectorizer (vectorizer) using the classifier model given (classifier); returns the predicted action label."""
    chat_in = []
    chat_in.append(chat_input)
    chat_in = vectorizer.transform(np.array(chat_in))

    return classifier.predict(chat_in)


# testing and evaluation
def test_baseline2(x_test, y_test):
    """Evaluates the performance of baseline2 classifier. Takes as input the test utterances (x_test) and corresponding
    action labels (y_test); returns the accuracy of the baseline as a string."""

    print(type(x_test), ":", type(y_test))

    tp = 0
    for i in range(len(y_test)):
        if predict_baseline2(x_test[i]) == y_test[i]:  # if the predicted label is the same as the actual
            tp += 1  # increment true positives

    return str(round(tp/len(y_test)*100, 2)) + "%"


def print_prediction_report(y_test, y_pred):
    """Prints evaluation metrics for a given classifier prediction, comparing the predicted action labels (y_pred) with
    the actual ones (y_test)"""
    print("predictions:")
    print(y_pred)
    print("performance reports:")
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))


def test_classifier(classifier, x_test, y_test):
    """Prints evaluation metrics for a given classifier. Takes as input the classifier to be evaluated (classifier), the
    test utterances (x_test) and corresponding action labels (y_test)."""
    y_pred = classifier.predict(x_test)
    print_prediction_report(y_test, y_pred)

