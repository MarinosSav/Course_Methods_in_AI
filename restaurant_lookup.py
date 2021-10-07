# restauraunt lookup

import random

def find_matching_restaurants(restaurants_data, pricerange, food, area):

    matches = []
    for restaurant in restaurants_data:
        if (    (pricerange == "any" or pricerange == restaurant['pricerange'])
            and (food == "any" or food == restaurant['food'])
            and (area == "any" or area == restaurant['area'])):

            matches.append(restaurant)

    return matches

# this will store the matches/alternatives in some variables to use later for the info when we connect to the rest of the code. for now i'm just printing them
def choose_restaurant(matches):

    selected_restaurant = random.choice(matches)
    alternatives = matches.copy()
    alternatives.remove(selected_restaurant)

    return selected_restaurant['restaurantname'], alternatives