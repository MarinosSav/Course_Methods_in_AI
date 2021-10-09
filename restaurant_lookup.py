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


def choose_restaurant(matches):

    selected_restaurant = random.choice(matches)
    alternatives = matches.copy()
    alternatives.remove(selected_restaurant)

    return selected_restaurant['restaurantname'], alternatives