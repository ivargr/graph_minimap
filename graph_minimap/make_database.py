import sqlite3

minimizer_db = sqlite3.connect('minimizers.db')
c = minimizer_db.cursor()
c.execute("create table minimizers (minimizer_hash int, chromosome int, linear_offset int, node int, offset int);")
c.execute("create index minimizer_hash on minimizers (minimizer_hash);")
c.execute("create unique index pos on minimizers (minimizer_hash, node, offset);")

minimizer_db.commit()
c.close()
