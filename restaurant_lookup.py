# restauraunt lookup

import random

def find_matching_restaurants(restaurants_data, pricerange, food, area):
    """Creates a list of restaurants that match given criteria. Receives as input the list of all restaurants
    (restaurants_data) and the criteria for the price range (pricerange), food choice (food) and area (area); outputs
    a list of potential matches (matches)"""
    matches = []
    for restaurant in restaurants_data:
        if (    (pricerange == "any" or pricerange == restaurant['pricerange'])
            and (food == "any" or food == restaurant['food'])
            and (area == "any" or area == restaurant['area'])):

            matches.append(restaurant)

    return matches


def choose_restaurant(matches):
    """Chooses one restaurant out of a list of potential restaurants (matches); outputs the restaurant
    (selected_restaurant) as well as the remaining restaurants in a separate list (alternatives)."""
    selected_restaurant = random.choice(matches)
    alternatives = matches.copy()
    alternatives.remove(selected_restaurant)

    return selected_restaurant, alternatives