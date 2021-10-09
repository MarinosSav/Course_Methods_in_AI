import re
import nltk
import random

# Regex patterns for each option
FOOD_PATTERN = r"(\w+)(\sfood|\srestaurant)"
AREA_PATTERN = r"in the (\w+)"
PRICE_PATTERN = r"(\w+)(\spriced|\srestaurant)"
MAX_ALLOWED_DISTANCE = 3  # maximum levenshtein distance


def find_nearest_option(word, options):
    """Returns the closest distance Levenshtein word (word) from a set of options (options)."""
    closest_options = []
    current_best_distance = 999
    for option in options:
        distance = nltk.edit_distance(word, option)
        if distance <= MAX_ALLOWED_DISTANCE:
            if distance < current_best_distance:
                closest_options = [option]
            elif distance == current_best_distance:
                closest_options.append(option)
    if closest_options:
        return random.choice(closest_options)
    return None


def get_preferences(utterance, food_options, area_options, price_options):
    """Extracts the preferences of the user for food type (food_match), area (area_match) and price range (price_match)
    from a utterance (utterance). Takes as input the option set for each."""
    food_match, area_match, price_match = check_exact_matches(utterance, food_options, area_options, price_options)
    if food_match:
        food_match = food_match[0]
    else:
        food_match = check_pattern_matches(utterance, FOOD_PATTERN, food_options)
    if area_match:
        area_match = area_match[0]
    else:
        area_match = check_pattern_matches(utterance, AREA_PATTERN, area_options)
    if price_match:
        price_match = price_match[0]
    else:
        price_match = check_pattern_matches(utterance, PRICE_PATTERN, price_options)


    return food_match, area_match, price_match


def check_pattern_matches(utterance, pattern, options):
    """Checks if a given word pattern (pattern) is detected within a utterance (utterance), before attempting to find
    the closes possible word. If no word is found None is returned."""
    match = re.search(pattern, utterance)
    if match:
        word = match[1]
        if word == "any":
            return "any"
        return find_nearest_option(word, options)
    return None


def check_exact_matches(utterance, food_options, area_options, price_options):
    """Checks utterance (utterance) for exact matches from the options sets for food type (food_options), area
    (area_options) and price range (price_options)."""
    food_match = [ele for ele in food_options if(ele in utterance)]
    area_match = [ele for ele in area_options if(ele in utterance)]
    price_match = [ele for ele in price_options if(ele in utterance)]

    return food_match, area_match, price_match
