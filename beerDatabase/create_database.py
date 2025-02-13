import json
import sqlite3

# Load JSON data from file
with open('/Users/Raneet/Desktop/iBeer.ai/iBeer.ai/beerDatabase/dataset.json', 'r', encoding='utf-8') as file:
    beers = json.load(file)

# Connect to SQLite database (Creates a new one if it doesn't exist)
conn = sqlite3.connect('beer_database.db')
cursor = conn.cursor()

# Create table structure
cursor.execute('''
    CREATE TABLE IF NOT EXISTS beers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name_of_beer TEXT,
        history_and_background TEXT,
        water TEXT,
        malt TEXT,
        hops TEXT,
        yeast TEXT,
        mashing TEXT,
        boiling TEXT,
        cooling TEXT,
        fermentation TEXT,
        conditioning TEXT,
        filtering_and_packaging TEXT,
        note TEXT
    )
''')

# Create a table for food pairing
cursor.execute('''
    CREATE TABLE IF NOT EXISTS food_pairing (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        beer_id INTEGER,
        food TEXT,
        FOREIGN KEY (beer_id) REFERENCES beers(id)
    )
''')

# Create a table for smokes pairing
cursor.execute('''
    CREATE TABLE IF NOT EXISTS smokes_pairing (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        beer_id INTEGER,
        smoke TEXT,
        FOREIGN KEY (beer_id) REFERENCES beers(id)
    )
''')

# Insert data into beers table
for beer in beers:
    cursor.execute('''
        INSERT INTO beers (name_of_beer, history_and_background, water, malt, hops, yeast,
                           mashing, boiling, cooling, fermentation, conditioning, filtering_and_packaging, note)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (beer["name_of_beer"], beer["history_and_background"], beer["ingredients"]["water"], beer["ingredients"]["malt"],
          beer["ingredients"]["hops"], beer["ingredients"]["yeast"], beer["brewing_process"]["mashing"],
          beer["brewing_process"]["boiling"], beer["brewing_process"]["cooling"], beer["brewing_process"]["fermentation"],
          beer["brewing_process"].get("conditioning", "N/A"),  
          beer["brewing_process"]["filtering_and_packaging"], beer["note"]))

    # Get the last inserted beer ID
    beer_id = cursor.lastrowid

    # Insert food pairing data
    for food in beer["food_pairing"]:
        cursor.execute("INSERT INTO food_pairing (beer_id, food) VALUES (?, ?)", (beer_id, food))

    # Insert smokes pairing data
    for smoke in beer["smokes"]:
        cursor.execute("INSERT INTO smokes_pairing (beer_id, smoke) VALUES (?, ?)", (beer_id, smoke))

# Commit changes and close connection
conn.commit()
conn.close()

print("Database successfully created and populated!")
