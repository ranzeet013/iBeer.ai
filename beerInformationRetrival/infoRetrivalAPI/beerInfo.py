import json
from fuzzywuzzy import process

# Load the beer data
with open('C:/Users/suj33/Desktop/iBeer.ai/beerInformationRetrival/infoRetrivalAPI/beerInformatonDataset/data.json') as f:
    beers_data = json.load(f)

def get_best_match(input_name: str, beers_list: list):
    names = [beer['name_of_beer'] for beer in beers_list]
    best_match = process.extractOne(input_name, names)
    return best_match
