import sqlite3

conn = sqlite3.connect('beer_database.db')
cursor = conn.cursor()

cursor.execute("PRAGMA table_info(beers);")
columns = cursor.fetchall()

print("\nColumns in 'beers' table:")
for col in columns:
    print(col)

conn.close()
