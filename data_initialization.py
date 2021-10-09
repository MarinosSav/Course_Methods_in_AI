from csv import DictReader
from reasoning import add_extra_restaurant_properties

# stored files
RESTAURANT_INFO_FILENAME = 'restaurant_info_new.csv'
DIALOG_DATASET_FILENAME = 'dialog_acts.dat'


def fetch_sample_dialogs_data():
    """Reads dialog input file and splits data into two lists, one for utterances and one for dialog actions; returns
    these lists."""
    utterances = []
    actions = []
    with open(DIALOG_DATASET_FILENAME) as fp:
        for line in fp:
            item = line.rstrip()  # Remove trailing characters
            action = item.split(' ', 1)[0]  # Select the action by choosing the first word
            utterance = item.replace(action, "")  # Select the utterance by removing the action for the line
            utterances.append(utterance.lower())
            actions.append(action.lower())
    return utterances, actions


def fetch_restaurant_info():
    """Reads the file containing restaurant information and writes it as a list."""
    with open(RESTAURANT_INFO_FILENAME, 'r') as read_obj:
        dict_reader = DictReader(read_obj)
        restaurant_list = list(dict_reader)
    return restaurant_list


def initialize_restaurant_info():
    """Loads the restaurant information add additional fields."""
    restaurant_info = fetch_restaurant_info()
    updated_restaurant_info = add_extra_restaurant_properties(restaurant_info)
    return updated_restaurant_info


def get_category_options(restaurants_info):
    """Creates sets for each potential option a user can choose for each restaurant and returns them; one for food
    (food_options), one for area (area_options) and one for price range options (pricerange_options)."""
    food_options = set([d['food'] for d in restaurants_info if 'food' in d])
    area_options = set([d['area'] for d in restaurants_info if 'area' in d])
    pricerange_options = set([d['pricerange'] for d in restaurants_info if 'pricerange' in d])

    # area_options.remove('')

    return food_options, area_options, pricerange_options