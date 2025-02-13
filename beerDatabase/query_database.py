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
