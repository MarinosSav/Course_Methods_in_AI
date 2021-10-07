import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.metrics import classification_report, accuracy_score

# ML training + baselines
def predict_baseline1(baseline1, chat_input):
    return baseline1.predict(chat_input)[0]

def fit_baseline1(x_train, y_train):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    return dummy

def baseline2(chat_input):
    """Baseline 2"""

    rules = [("thankyou", ["thank you", "thanks"]),
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

    for rule in rules:
        for keyword in rule[1]:
            if keyword in chat_input:
                return rule[0]

    # default return value, for all expressions not found in the rules list
    return "inform"

def fit_tree_classifier(x_train, y_train):
    clf = tree.DecisionTreeClassifier()
    clf.fit(x_train, y_train)

    return clf

def fit_lr_classifier(x_train, y_train):
    lr = LogisticRegression()
    lr.fit(x_train, y_train)

    return lr

def predict(classifier, vectorizer, chat_input):
    chat_in = []
    chat_in.append(chat_input)
    chat_in = vectorizer.transform(np.array(chat_in))

    return classifier.predict(chat_in)

# testing and evaluation
def test_baseline2(x_test, y_test):
    """Evaluates performance of baseline2"""

    tp = 0
    for i in range(len(y_test)):
        if baseline2(x_test[i]) == y_test[i]:
            tp += 1
        else:
            pass
            #print(baseline2(x_test[i]), ":", y_test[i], x_test[i])

    return str(round(tp/len(y_test)*100, 2)) + "%"

def print_prediction_report(y_test, y_pred):
    print("predictions:")
    print(y_pred)
    print("performance reports:")
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))

def test_tree_classifier(classifier, x_test, y_test, label_encoder):
    pred_result = classifier.predict(x_test)
    y_pred = label_encoder.inverse_transform(pred_result)
    print_prediction_report(y_test, y_pred)

def test_lr_classifier(classifier, x_test, y_test):
    y_pred = classifier.predict(x_test)
    print_prediction_report(y_test, y_pred)
