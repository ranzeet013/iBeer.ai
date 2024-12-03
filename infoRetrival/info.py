from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fuzzywuzzy import process
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow any origin (you can limit to specific domains)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the beer data (replace with loading from your actual JSON file)
with open('/Users/Raneet/Desktop/IBeer.ai/infoRetrival/data.json') as f:
    beers_data = json.load(f)

# Beer model (this will match the structure of your beer data)
class Beer(BaseModel):
    name_of_beer: str
    history_and_background: str
    ingredients: dict
    brewing_process: dict
    similar_beer: list
    food_pairing: str

# Utility function to perform fuzzy search on beer names
def get_best_match(input_name: str, beers_list: list):
    # Get the best matching beer name using fuzzywuzzy's process.extractOne
    names = [beer['name_of_beer'] for beer in beers_list]
    best_match = process.extractOne(input_name, names)
    return best_match

@app.get("/beer/{beer_name}", response_model=Beer)
async def get_beer_info(beer_name: str):
    # Perform fuzzy search to find the best matching beer name
    best_match = get_best_match(beer_name, beers_data)
    if best_match and best_match[1] > 80:  # If confidence is greater than 80%
        matched_beer_name = best_match[0]
        # Find the beer data corresponding to the matched name
        for beer in beers_data:
            if beer['name_of_beer'] == matched_beer_name:
                return beer
    else:
        return {"error": "No similar beer found. Please check the name."}

@app.get("/")
async def root():
    return {"message": "Welcome to the Beer Info API! Use '/beer/{beer_name}' to get beer details."}
