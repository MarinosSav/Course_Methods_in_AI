from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from restaurant_lookup import find_matching_restaurants, choose_restaurant
from preference_extraction import get_preferences, find_nearest_option, set_max_distance
from classifiers import predict_baseline1, predict_baseline2, test_baseline2, test_classifier, fit_classifier, predict
from data_initialization import fetch_sample_dialogs_data, get_category_options, initialize_restaurant_info
from reasoning import get_additional, choose_with_extra_reqs

# TODO: Add config
# TODO: Add README

# outputs to be used by the chatbot
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
                     'nopref': 'You must first choose your preferred restaurant.',
                     'additional': 'Do you have any additional requirements?'}

# chatbot class
class RestaurantChatbot:
    def __init__(self):
        self.restaurants_info = initialize_restaurant_info()
        self.food_options, self.area_options, self.price_options = get_category_options(self.restaurants_info)
        self.mode = 3
        self.state = None
        self.preference = {'restaurantname': None, 'food': None, 'area': None, 'pricerange': None}
        self.recommendation = []
        self.alternatives = []
        self.request = []
        self.debug_mode = 1
        self.previous_state = None
        print("Chatbot initiated...\nDefault classifier: Logistic Regression\n\n" + SYSTEM_UTTERANCES['welcome'])

    def configure(self, config, config_value):

        HELP = ""

        try:
            if config == "/h":  # prints list of commands to be used
                print(HELP)
            if config == "/d":  # debug mode on/off
                if config_value == "on":
                    self.debug_mode = 1
                elif config_value == "off":
                    self.debug_mode = 0
            elif config == "/l":  # change levenshtein distance
                set_max_distance(config_value)
            elif config == "/c":  # check for correctness check on/off
                pass
            elif config == "/r":  # allow dialog restarts on/off
                pass
            elif config == "/d":  # system delay before response
                pass
            elif config == "/o":  # output in all caps on/off
                pass
            elif config == "/s":  # classifier switching
                if config_value == "b1":
                    self.mode = 1
                    print("Switched to baseline 1")
                elif config_value == "b2":
                    self.mode = 2
                    print("Switched to baseline 2")
                elif config_value == "c1":
                    self.mode = 3
                    print("Switched to logistic regression classifier")
                elif config_value == "c2":
                    self.mode = 4
                    print("Switched to tree classifier")
        except:
            print("Error: Configuration command not recognized")

    def run_chatbot(self, x_test, y_test, count_vect, baseline1, tree_classifier, logistic_classifier):
        """The core of the chatbot, handles parsing and calling the functions. Takes as input the test utterances
        (x_test), the test action labels (y_test), the fitted vectorizer (count_vect) as well as all the classifier
        models for the dummy (baseline1), tree (tree_classifier) and logistic regression (logistic_classifier)
        classifiers."""

        while True:
            chat_input = input(">").lower()
            if chat_input.startswith("/"):
                line = chat_input.split(' ', 1)
                if len(line) > 1:
                    cmd_value = line[1]
                else:
                    cmd_value = None
                cmd = line[0]
                if cmd == "/e":  # exit the program
                    break
                elif cmd == "/t":  # test the classifiers
                    if cmd_value == "b1":
                        pass  # TODO: Implement for this
                    elif cmd_value == "b2":
                        test_baseline2(x_test, y_test)
                    elif cmd_value == "c1":
                        test_classifier(logistic_classifier, x_test, y_test)
                    elif cmd_value == "c2":
                        test_classifier(tree_classifier, x_test, y_test)
                    else:
                        pass
                else:
                    self.configure(cmd, cmd_value)
            else:
                if self.mode == 1:
                    predicted_label = predict_baseline1(baseline1, chat_input)
                elif self.mode == 2:
                    predicted_label = predict_baseline2(chat_input)
                elif self.mode == 3:
                    predicted_label = predict(logistic_classifier, count_vect, chat_input)
                else:
                    predicted_label = predict(tree_classifier, count_vect, chat_input)
                if self.debug_mode:
                    print("input:", chat_input, "->", predicted_label)
                self.previous_state = self.state
                self.state = self.state_transition(predicted_label, chat_input)
                if self.debug_mode:
                    print("state:", self.previous_state, "->", self.state)
                self.preference, self.recommendation, self.alternatives = dialog_manager(self.restaurants_info, self.state, self.preference, self.recommendation, self.alternatives,
                                                                        self.request)
                if self.debug_mode:
                    print("preference:", self.preference)

                #records.append((new_action, chat_input))
                #print(new_action)

    def state_transition(self, predicted_label, chat_input):
        """The state transition handler. Returns the next state given the predicted action label (predicted_lable) and
        user chat input (chat_input)."""
        if predicted_label == 'inform' or predicted_label == 'reqalts':  # TODO: Make 'Any' an option
            preferences = get_preferences(chat_input, self.food_options, self.area_options, self.price_options)
            if preferences[0]:
                self.preference['food'] = preferences[0]
            if preferences[1]:
                self.preference['area'] = preferences[1]
            if preferences[2]:
                self.preference['pricerange'] = preferences[2]

            if self.preference['food'] and self.preference['area'] and self.preference['pricerange']:
                if self.previous_state == 'request_additional':
                    next_state = 'suggest_additional'
                elif self.previous_state == 'suggest_additional':
                    next_state = 'restaurant_suggestion'
                else:
                    next_state = 'request_additional'
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


# potential requests
REQUESTS = [("address", ["address"]),
            ("tel_nr", ["phone number", "number", "telephone number"]),
            ("food", ["type of food", "food"]),
            ("postal_code", ["post code", "postcode", "postal code"]),
            ("price", ["price range", "how much", "price"]),
            ("area", ["area", "located", "part of town", "where"])]
def get_request(user_input):
    """Extracts what the user is looking for in a request e.g. phone number, address. Takes as input the user chat input
    (user_input) and outputs a list of all items requested (requests)."""
    requests = []
    for request in REQUESTS:
        for keyword in request[1]:
            if keyword in user_input:
                requests.append(keyword[0])

    return requests


# dialog manager
def dialog_manager(restaurants_info, state, preference, recommendation, alternatives, chat_input):
    """Function handling majority of dialog. Takes as input the list with all the restaurant information
    (restaurant_info), the current state (state) and user input (chat input); updating the preference tuple
    (preference) as well as the current recommendation list (recommendation) and list of alternative restaurants
    (alternatives)"""

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
    elif state == 'request_additional':
        print(SYSTEM_UTTERANCES['additional'])
    elif state == 'suggest_additional':
        additional_requirements = get_additional(chat_input)
        matches = find_matching_restaurants(restaurants_info, preference['pricerange'], preference['food'],
                                            preference['area'])
        additional_matches = choose_with_extra_reqs(matches, additional_requirements)
        if additional_matches:
            restaurant, alternatives = choose_restaurant(matches)
            preference['restaurantname'] = restaurant
            print(SYSTEM_UTTERANCES['restaurant_suggestion'] % (restaurant, preference['food'], preference['area']))
        else:
            print(SYSTEM_UTTERANCES['nomatch'])
    elif state == 'confirm':
        #extract preferences  # TODO: Work on Confirm and Negate for Clarify
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

    print("Initiating...")
    x_data, y_data = fetch_sample_dialogs_data()

    # preprocess data
    count_vect = CountVectorizer()  #stop_words = 'english'
    x_bag = count_vect.fit_transform(x_data)

    # split data into training and testing data
    x_train, x_test, y_train, y_test = train_test_split(x_bag, y_data, test_size=0.15)

    # create classifiers
    baseline1 = fit_classifier(x_train, y_train)
    tree_classifier = fit_classifier(x_train, y_train, "tree")
    logistic_classifier = fit_classifier(x_train, y_train, "lr")

    # initiate and run the chatbot
    chatbot = RestaurantChatbot()
    chatbot.run_chatbot(x_test, y_test, count_vect, baseline1, tree_classifier, logistic_classifier)


main()
