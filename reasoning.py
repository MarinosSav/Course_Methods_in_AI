from restaurant_lookup import choose_restaurant

INFERENCE_RULES = [{"ID": 1, "antecedent": {"pricerange": "cheap", "good food": True}, "consequent": "busy", "value": True, "description": "a cheap restaurant with good food is busy"},
                   {"ID": 2, "antecedent": {"food": "spanish"}, "consequent": "long stay", "value": True,"description": "Spanish restaurants serve extensive dinners that take a long time to finish"},
                   {"ID": 3, "antecedent": {"busy": True}, "consequent": "long stay", "value": True, "description": "you spend a long time in a busy restaurant (waiting for food)"},
                   {"ID": 4, "antecedent": {"long stay": True}, "consequent": "children", "value": False, "description": "spending a long time is not advised when taking children"},
                   # {"ID": 5, "antecedent": {"crowdedness":"busy"}, "consequent": "romantic", "value": False, "description": "a busy restaurant is not romantic"},
                   {"ID": 6, "antecedent": {"long stay": True}, "consequent": "romantic", "value": True,"description": "spending a long time in a restaurant is romantic"}]


def add_extra_restaurant_properties(all_restaurant_info):
    """Goes over the inference rules table and the restaurants data received from the file, uses the reasoning logic to
    add all extra inferred properties to the restaurants dataset used for the chatbot."""
    for restaurant_info in all_restaurant_info:
        restaurant_info["inferred_properties"] = {}
        restaurant_info["inference_ids"] = []
        infer_extra_properties(restaurant_info)
    return all_restaurant_info


def check_inference_rule_match(restaurant_info, antecedent):
    """Checks for a match between a specific inference rule and a restaurant."""
    rule_matches = True
    for key, value in antecedent.items():
        if str(restaurant_info[key]).lower() != str(value).lower():
            rule_matches = False
            break
    return rule_matches


def infer_extra_properties(restaurant_info):
    """Using the inference rules table, reasons all the extra properties for a specific restaurant."""
    new_inference_made = True  # for first entry into the loop
    # need to keep checking for inferences every time a new one is made/found, as it could lead to new consequents/inferences
    while new_inference_made:
        new_inference_made = False
        for rule in INFERENCE_RULES:
            # if restaurant matches antecedent logic and this inference rule hasn't already been added
            # (necessary to avoid endless loops of two rules setting the same consequent of another to true/false alternatively)
            if check_inference_rule_match(restaurant_info, rule["antecedent"]) and not rule["ID"] in restaurant_info["inference_ids"]:
                restaurant_info["inferred_properties"][rule["consequent"]] = {"value": rule["value"], "description": rule["description"]}
                restaurant_info[rule["consequent"]] = rule["value"]
                restaurant_info["inference_ids"].append(rule["ID"])
                new_inference_made = True


NEGATION_KEYWORD = "not"
def get_additional(user_input):
    """Extracts what the user is looking for in an additional request. Takes as input the user chat input (user_input)
    and outputs a list of all items requested (additional)."""
    possible_additional_requirements = set([d['consequent'] for d in INFERENCE_RULES if 'consequent' in d])
    additional_requirement = None
    for keyword in possible_additional_requirements:
        if keyword in user_input:
            required_value = NEGATION_KEYWORD not in user_input
            additional_requirement = {"property": keyword, "value": required_value}
            break
    return additional_requirement


def choose_with_extra_reqs(candidates, req):
    """Chooses a restaurant from a list of candidates, according to a match between it's inferred properties
    and an extra user requirement passed as a parameter."""
    matches = [d for d in candidates if req['property'] in d['inferred_properties'] and
               d['inferred_properties'][req['property']]['value'] == req['value']]
    return matches
