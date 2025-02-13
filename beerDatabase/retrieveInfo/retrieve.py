from fastapi import FastAPI, HTTPException
import sqlite3
from rapidfuzz import fuzz

app = FastAPI()

def get_db_connection():
    conn = sqlite3.connect("/Users/Raneet/Desktop/iBeer.ai/iBeer.ai/beerDatabase/beer_database.db")
    conn.row_factory = sqlite3.Row 
    return conn

@app.get("/beer/{beer_name}")
def get_beer_info(beer_name: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name_of_beer FROM beers")
    beers = cursor.fetchall()
    
    best_match = None
    best_score = 0
    
    for beer in beers:
        score = fuzz.ratio(beer_name.lower(), beer["name_of_beer"].lower())
        if score > best_score and score >= 80:
            best_score = score
            best_match = beer
    
    if not best_match:
        conn.close()
        raise HTTPException(status_code=404, detail="Beer not found")
    
    beer_id = best_match["id"]
    cursor.execute("""
        SELECT name_of_beer, history_and_background, water, malt, hops, yeast, 
               mashing, boiling, cooling, fermentation, conditioning, filtering_and_packaging, note
        FROM beers WHERE id = ?
    """, (beer_id,))
    beer_info = cursor.fetchone()

    cursor.execute("SELECT food FROM food_pairing WHERE beer_id = ?", (beer_id,))
    foods = [food["food"] for food in cursor.fetchall()]

    cursor.execute("SELECT smoke FROM smokes_pairing WHERE beer_id = ?", (beer_id,))
    smokes = [smoke["smoke"] for smoke in cursor.fetchall()]
    
    conn.close()
    
    return {
        "name": beer_info["name_of_beer"],
        "history_and_background": beer_info["history_and_background"],
        "water": beer_info["water"],
        "malt": beer_info["malt"],
        "hops": beer_info["hops"],
        "yeast": beer_info["yeast"],
        "mashing": beer_info["mashing"],
        "boiling": beer_info["boiling"],
        "cooling": beer_info["cooling"],
        "fermentation": beer_info["fermentation"],
        "conditioning": beer_info["conditioning"],
        "filtering_and_packaging": beer_info["filtering_and_packaging"],
        "note": beer_info["note"],
        "food_pairings": foods,
        "smoke_pairings": smokes
    }

# Run the API using uvicorn
# Command: uvicorn filename:app --reload
