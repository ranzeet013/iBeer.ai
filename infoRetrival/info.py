from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fuzzywuzzy import process
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('/Users/Raneet/Desktop/IBeer.ai/infoRetrival/data.json') as f:
    beers_data = json.load(f)

# Beer model
class Beer(BaseModel):
    name_of_beer: str
    history_and_background: str
    ingredients: dict
    brewing_process: dict
    note: str
    food_pairing: list[str]
    smokes: list[str]

# Utility function to perform fuzzy search on beer names
def get_best_match(input_name: str, beers_list: list):
    names = [beer['name_of_beer'] for beer in beers_list]
    best_match = process.extractOne(input_name, names)
    return best_match

@app.get("/beer/{beer_name}", response_model=Beer)
async def get_beer_info(beer_name: str):
    best_match = get_best_match(beer_name, beers_data)
    if best_match and best_match[1] > 80: 
        matched_beer_name = best_match[0]
        for beer in beers_data:
            if beer['name_of_beer'] == matched_beer_name:
                return {
                    "name_of_beer": beer["name_of_beer"],
                    "history_and_background": beer["history_and_background"],
                    "ingredients": beer["ingredients"],
                    "brewing_process": beer["brewing_process"],
                    "note": beer["note"],
                    "food_pairing": beer["food_pairing"],
                    "smokes": beer["smokes"]
                }
    else:
        return {"error": "No similar beer found. Please check the name."}

@app.get("/")
async def root():
    return {"message": "Welcome to the Beer Info API! Use '/beer/{beer_name}' to get beer details."}
