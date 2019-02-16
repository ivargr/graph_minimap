import sqlite3

db = sqlite3.connect("minimizers_chr6.db")
c = db.cursor()

for hit in c.execute("SELECT * FROM minimizers where minimizer_hash=119209289550781").fetchall():
    print(hit)

print("Done")
