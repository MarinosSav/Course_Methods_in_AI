import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import preprocessing
from sklearn.metrics import classification_report, accuracy_score
from csv import DictReader
from restaurant_lookup import find_matching_restaurants, choose_restaurant
from preference_extraction import get_preferences, find_nearest_option

# constants
SYSTEM_UTTERANCES = {'welcome': 'Hello! Welcome to the UU restaurant system. How may I help you?',
                     'preferences': {'food_type': 'What kind of food do you have in mind?',
                                     'area': 'In which area do you please to eat?',
                                     'price_range': 'Do you wish to eat in the cheap, moderate of expensive price range?'},
                     'restaurant_suggestion': '%s is a fantastic %s restaurant in the %s of town.',
                     'provide_info': {'address': 'The address of %s is %s',
                                      'tel_nr': 'The telephone number of %s is %s',
                                      'postal_code': 'The postalcode of %s is %s',
                                      'food': '%s serves %s food',
                                      'area': '%s is in the %s of town',
                                      'price': 'The price range of %s is %s'},
                     'restart_convo': 'Welcome again! How can I help you this time?',
                     'alternative': 'Would you like to eat at %s in the %s of town, their prices are %s',
                     'alt_preferences': 'There are no restaurants available with your preferences, would you like to change them?',
                     'goodbye': 'Goodbye! And thank you for using the UU restaurant system',
                     'clarify': 'I did not quite understand that, did you mean %s?',
                     'nomatch': 'Unfortunately we did not find a restaurant matching your description.',
                     'nopref': 'You must first choose your preferred restaurant.'}

# data initialization
def fetch_data():
    """Reads input file and splits into utterances and dialog actions"""
    utterances = []
    actions = []
    with open('dialog_acts.dat') as fp:
        for line in fp:
            item = line.rstrip()  # Remove trailing characters
            action = item.split(' ', 1)[0]  # Select the action by choosing the first word
            utterance = item.replace(action, "")  # Select the utterance by removing the action for the line
            utterances.append(utterance.lower())
            actions.append(action.lower())
    return utterances, actions

def fetch_restaurant_info():
    with open('restaurant_info.csv', 'r') as read_obj:
        dict_reader = DictReader(read_obj)
        restaurant_list = list(dict_reader)
    return restaurant_list

# will store these in variables to by used in the preference matching
def initialize_category_options(restaurants_info):
    food_options = set([d['food'] for d in restaurants_info if 'food' in d])
    area_options = set([d['area'] for d in restaurants_info if 'area' in d])
    pricerange_options = set([d['pricerange'] for d in restaurants_info if 'pricerange' in d])

    # area_options.remove('')

    return food_options, area_options, pricerange_options

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

# chatbot main
class RestaurantChatbot:
    def __init__(self):
        self.restaurants_info = fetch_restaurant_info()
        self.food_options, self.area_options, self.price_options = initialize_category_options(self.restaurants_info)
        self.mode = 3
        self.state = None
        self.preference = {'restaurantname': None, 'food': None, 'area': None, 'pricerange': None}
        self.recommendation = []
        self.alternatives = []
        self.request = []
        print("Chatbot initiated...\nDefault classifier: Logistic Regression\n\n" + SYSTEM_UTTERANCES['welcome'])

    def run_chatbot(self, x_test, y_test, count_vect, label_encoder, baseline1, tree_classifier, logistic_classifier):
        while True:
            chat_input = input(">").lower()
            if chat_input == "/e":
                break
            elif chat_input.split(' ', 1)[0] == "/t":
                test = chat_input.split(' ', 1)[1]
                if test == "b1":
                    pass
                elif test == "b2":
                    test_baseline2(x_test, y_test)
                elif test == "c1":
                    test_lr_classifier(logistic_classifier, x_test, y_test)
                elif test == "c2":
                    test_tree_classifier(tree_classifier, x_test, y_test, label_encoder)
                else:
                    pass
            elif chat_input == "/b1":
                self.mode = 1
                print("Switched to baseline 1")
            elif chat_input == "/b2":
                self.mode = 2
                print("Switched to baseline 2")
            elif chat_input == "/c1":
                self.mode = 3
                print("Switched to logistic regression classifier")
            elif chat_input == "/c2":
                self.mode = 4
                print("Switched to tree classifier")
            else:
                if self.mode == 1:
                    predicted_label = predict_baseline1(baseline1, chat_input)
                elif self.mode == 2:
                    predicted_label = baseline2(chat_input)
                elif self.mode == 3:
                    predicted_label = predict(logistic_classifier, count_vect, chat_input)
                else:
                    predicted_label = predict(tree_classifier, count_vect, chat_input)
                print("input:", chat_input, "->", predicted_label)
                before_state = self.state
                self.state = self.state_transition(predicted_label, chat_input)
                print("state:", before_state, "->", self.state)
                self.preference, self.recommendation, self.alternatives = dialog_manager(self.restaurants_info, self.state, self.preference, self.recommendation, self.alternatives,
                                                                        self.request)
                print("preference:", self.preference)
                #records.append((new_action, chat_input))
                #print(new_action)

    def state_transition(self, predicted_label, chat_input):
        if predicted_label == 'inform' or predicted_label == 'reqalts':
            preferences = get_preferences(chat_input, self.food_options, self.area_options, self.price_options)
            if preferences[0]:
                self.preference['food'] = preferences[0]
            if preferences[1]:
                self.preference['area'] = preferences[1]
            if preferences[2]:
                self.preference['pricerange'] = preferences[2]

            if self.preference['food'] and self.preference['area'] and self.preference['pricerange']:
                next_state = 'restaurant_suggestion'
            else:
                next_state = 'request_info'
        elif predicted_label == 'reqmore' or predicted_label == 'deny' or predicted_label == 'negate':
            next_state = 'alt_preferences'
        elif predicted_label == 'repeat':
            next_state = self.state # keep current state
        elif predicted_label == 'hello':
            next_state = 'welcome'
        elif predicted_label == 'null':
            next_state = 'clarify'
        elif predicted_label == 'thankyou' or predicted_label == 'bye':
            next_state = 'goodbye'
        elif predicted_label == 'restart':
            next_state = 'restart_convo'
        elif predicted_label == 'request':
            next_state = 'provide_info'
        elif predicted_label == 'confirm':
            next_state = 'confirm' #need to extract preferences for this
        elif predicted_label == 'ack' or predicted_label == 'affirm':
            next_state = ''
        else:
            next_state = 'welcome'

        return next_state

def get_request(user_input):
    return ['address', 'price']

# dialog manager
def dialog_manager(restaurants_info, state, preference, recommendation, alternatives, chat_input):
    if state == 'restaurant_suggestion':
        matches = find_matching_restaurants(restaurants_info, preference['pricerange'], preference['food'], preference['area'])
        if matches:
            restaurant, alternatives = choose_restaurant(matches)
            preference['restaurantname'] = restaurant
            print(SYSTEM_UTTERANCES['restaurant_suggestion'] % (restaurant, preference['food'], preference['area']))
        else:
            print(SYSTEM_UTTERANCES['nomatch'])
    elif state == 'welcome' or state == 'goodbye' or state == 'clarify':
        print(SYSTEM_UTTERANCES[state])
    elif state == 'clarify':
        print(SYSTEM_UTTERANCES[state] % (find_nearest_option(chat_input)))  # TODO: Extract only the word instead of whole input
    elif state == 'alt_preferences':
        if alternatives:
            restaurant, alternatives = choose_restaurant(alternatives)
            print(SYSTEM_UTTERANCES['restaurant_suggestion'] % (restaurant['restaurantname'],
                                                                restaurant['food'], restaurant['area']))
        else:
            print(SYSTEM_UTTERANCES['nomatch'])
    elif state == 'restart_convo':
        preference = {'restaurantname': None, 'food': None, 'area': None, 'pricerange': None}
        recommendation = []
        alternatives = []
    elif state == 'provide_info':
        if not preference['restaurantname']:
            print(SYSTEM_UTTERANCES['nopref'])
        else:
            for request in get_request(chat_input):
                print(SYSTEM_UTTERANCES['provide_info'][request] % (preference['restaurantname'], preference[request]))
    elif state == 'request_info':
        if not preference['food']:
            print(SYSTEM_UTTERANCES['preferences']['food_type'])
        elif not preference['area']:
            print(SYSTEM_UTTERANCES['preferences']['area'])
        else:
            print(SYSTEM_UTTERANCES['preferences']['price_range'])
    elif state == 'confirm':
        #extract preferences
        pass
    else:
        print("Error: state %s not found" % state)
        pass
        # print the question depending on preference variable
        # request = get_request(user_input) -> decide message according to preferences missing
        # get_preferences(utterance, food_options, area_options, price_options)

        # test_utterance = "I want expensive british food in the whatever"
        # food_match, area_match, price_match = get_preferences(test_utterance, food_options, area_options, price_options)
        # print(food_match)
        # print(area_match)
        # print(price_match)

    return preference, recommendation, alternatives

def main():

    print("Starting chatbot...")
    x_data, y_data = fetch_data()

    # preprocess data
    count_vect = CountVectorizer()
    x_bag = count_vect.fit_transform(x_data)

    x_train, x_test, y_train, y_test = train_test_split(x_bag, y_data, test_size=0.15)

    # encode labels for decision tree classifier
    le = preprocessing.LabelEncoder()
    le.fit(y_data)
    encoded_y_train = le.transform(y_train)
    baseline1 = fit_baseline1(x_train, y_train)

    # create and test classifiers
    tree_classifier = fit_tree_classifier(x_train, y_train)
    #test_tree_classifier(tree_classifer, x_test, y_test, le)
    logistic_classifier = fit_lr_classifier(x_train, y_train)
    #test_lr_classifier(logistic_classifier, x_test, y_test)

    chatbot = RestaurantChatbot()
    chatbot.run_chatbot(x_test, y_test, count_vect, le, baseline1, tree_classifier, logistic_classifier)

main()
