import sqlite3

conn = sqlite3.connect('beer_database.db')
cursor = conn.cursor()

cursor.execute("SELECT id, name_of_beer FROM beers")
beers = cursor.fetchall()

print("\nList of Beers:")
for beer in beers:
    print(f"{beer[0]}: {beer[1]}")

beer_name = "Tuborg Grøn (Green) Beer"

cursor.execute("SELECT id FROM beers WHERE name_of_beer = ?", (beer_name,))
beer_id = cursor.fetchone()

if beer_id:
    cursor.execute("""
        SELECT name_of_beer, history_and_background, water, malt, hops, yeast, 
               mashing, boiling, cooling, fermentation, conditioning, filtering_and_packaging, note
        FROM beers 
        WHERE id = ?
    """, (beer_id[0],))
    
    beer_info = cursor.fetchone()
    
    if beer_info:
        print("\nFull Beer Information:")
        print(f"Name: {beer_info[0]}")
        print(f"History and Background: {beer_info[1]}")
        print(f"Water: {beer_info[2]}")
        print(f"Malt: {beer_info[3]}")
        print(f"Hops: {beer_info[4]}")
        print(f"Yeast: {beer_info[5]}")
        print(f"Mashing: {beer_info[6]}")
        print(f"Boiling: {beer_info[7]}")
        print(f"Cooling: {beer_info[8]}")
        print(f"Fermentation: {beer_info[9]}")
        print(f"Conditioning: {beer_info[10]}")
        print(f"Filtering and Packaging: {beer_info[11]}")
        print(f"Note: {beer_info[12]}")
    

    cursor.execute("SELECT food FROM food_pairing WHERE beer_id = ?", (beer_id[0],))
    foods = cursor.fetchall()
    print("\nFood Pairings:")
    for food in foods:
        print("-", food[0])

    cursor.execute("SELECT smoke FROM smokes_pairing WHERE beer_id = ?", (beer_id[0],))
    smokes = cursor.fetchall()
    print("\nSmokes Pairings:")
    for smoke in smokes:
        print("-", smoke[0])

conn.close()
