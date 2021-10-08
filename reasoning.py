from restaurant_lookup import choose_restaurant

INFERENCE_RULES =  [{"ID": 1, "antecedent": {"pricerange": "cheap", "good food": True}, "consequent": "busy", "value": True, "description": "a cheap restaurant with good food is busy"},
                    {"ID": 2, "antecedent": {"food": "spanish"}, "consequent": "long stay", "value": True,"description": "Spanish restaurants serve extensive dinners that take a long time to finish"},
                    {"ID": 3, "antecedent": {"busy": True}, "consequent": "long stay","value": True, "description": "you spend a long time in a busy restaurant (waiting for food)"},
                    {"ID": 4, "antecedent": {"long stay": True}, "consequent": "children", "value": False, "description": "spending a long time is not advised when taking children"},
                    # {"ID": 5, "antecedent": {"crowdedness":"busy"}, "consequent": "romantic", "value": False, "description": "a busy restaurant is not romantic"},
                    {"ID": 6, "antecedent": {"long stay": True}, "consequent": "romantic", "value": True,"description": "spending a long time in a restaurant is romantic"}]

def add_extra_restaurant_properties(all_restaurant_info):
    for restaurant_info in all_restaurant_info:
        restaurant_info["extra_properties"] = {}
        find_new_inferences(restaurant_info)
    return all_restaurant_info

# this function might be replaced if i find a nicer equivalent for Ruby ".all" in python ;)
def check_inference_rule_match(restaurant_info, antecedent):
    rule_matches = True
    for key,value in antecedent.items():
        if restaurant_info[key] != value:
            rule_matches = False
            break
    return rule_matches

def consequent_already_added(restaurant_info, rule):
    return (rule["consequent"] in restaurant_info and restaurant_info[rule["consequent"]] == rule["value"])

def find_new_inferences(restaurant_info):
    new_inference_made = True # for first entry into the loop
    # need to keep checking for inferences every time a new one is made/found, as it could lead to new consequents/inferences
    while new_inference_made:
        new_inference_made = False
        for rule in INFERENCE_RULES:
            # if restaurant matches antecedent logic and consequent isn't already in the extra properties
            if check_inference_rule_match(restaurant_info, rule["antecedent"]) and not consequent_already_added(restaurant_info, rule):
                restaurant_info["extra_properties"] = {"property": rule["consequent"], "value": rule["value"], "description": rule["description"]}
                restaurant_info[rule["consequent"]] = rule["value"]
                new_inference_made = True

def get_additional_requirements():
    pass

def choose_with_extra_reqs(candidates, req):
    matches = [d for d in candidates if d['extra_properties'][req['property']] == req['value']]
    return choose_restaurant(matches)

