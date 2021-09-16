from sklearn.dummy import DummyClassifier
import numpy as np

"""Global Variables"""
records = []  # List to store all records of read file
training_set = []  # List to store the training partition
test_set = []  # List to store the test partition
mode = 1  # Variable indicating which baseline is being used


def initialize():
    """Reads input file and partitions data"""

    for line in open('dialog_acts.dat', 'r'):  # Read each line in filed
        item = line.rstrip()  # Remove trailing characters
        action = item.split(' ', 1)[0]  # Select the action by choosing the first word
        utterance = item.replace(action, "")  # Select the utterance by removing the action for the line
        records.append((action.lower(), utterance.lower()))  # Convert action and utterance to lowercase and store


def partition(size=0.85):
    """Partitions training data and test data"""

    for i in range(len(records)):
        if i < size * len(records):
            training_set.append(records[i])
        else:
            test_set.append(records[i])


def baseline1():
    """Baseline 1"""

    training_y = np.array(training_set)[:, 0]
    training_x = np.array(training_set)[:, 1]
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(training_x, training_y)
    action = dummy_clf.predict(chat_input)[0]

    return action


def baseline2():
    """Baseline 2"""

    rules = [("ack", ["okay", "thatll do", "alright", "kay", "good"]),
             ("affirm", ["yes", "yea", "uh huh"]),  # ye interferes with bye
             ("bye", ["bye", "good bye", "see you", "good night"]),
             ("confirm", ["is there", "is it", "does it", "is that", "do they"]),
             ("deny", ["dont", "not", "wrong", "change"]),
             ("hello", ["hi", "hey", "hello", "halo"]),
             ("negate", ["no"]),
             #("inform", ["looking", "searching", "trying to find", "cheap", "mediterranean", "asian", "bistro", "any"]),
             ("repeat", ["repeat", "go back", "come again"]),
             ("reqalts", ["how about", "what about", "next", "else", "another", "any other"]),
             ("reqmore", ["more"]),
             ("request", ["what", "where", "when", "could", "can", "address", "postcode", "phone number",
                          "type of food", "price range", "area"]),
             ("restart", ["star over", "reset", "start again"]),
             ("thankyou", ["thank you", "thanks"]),
             ("null", ["noise", "sil", "cough", "unintelligible", "noise", "tv_noise", "inaudible"])]

    for rule in rules:
        for keyword in rule[1]:
            if keyword in chat_input:
                return rule[0]

    return "inform"


def classifier1():
    """Classifier 1"""
    pass


def classifier2():
    """Classifier 2"""
    pass


initialize()
# Chat bot core
while True:
    partition()
    chat_input = input(">").lower()
    if chat_input == "/exit":
        break
    elif chat_input == "/b1":
        mode = 1
        print("Using baseline 1")
    elif chat_input == "/b2":
        mode = 2
        print("Using baseline 2")
    elif chat_input == "/c1":
        mode = 3
        print("Using classifier 1")
    elif chat_input == "/c2":
        mode = 4
        print("Using classifier 2")
    else:
        if mode == 1:
            new_action = baseline1()
        elif mode == 2:
            new_action = baseline2()
        elif mode == 3:
            new_action = classifier1()
        else:
            new_action = classifier2()
        records.append((new_action, chat_input))
        print(new_action)
