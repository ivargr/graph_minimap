import sys
import sqlite3
from graph_minimap.find_minimizers_in_kmers import make_databse

make_databse("all")
db = sqlite3.connect("minimizers_all.db")
c = db.cursor()

chromosomes = sys.argv[1].split(",")
for chromosome in chromosomes:
    print("Processing chromosome %s" % chromosome)
    c.execute('ATTACH DATABASE "minimizers_chr%s.db" AS chr%s' % (chromosome, chromosome))
    print("Attached")
    print("Deleting hashes with only Ns")
    # Remove hashes that are 0 (all Ns) to save some time
    c.execute("DELETE FROM chr%s.minimizers where minimizer_hash=0" % chromosome)
    db.commit()
    print("Copying")
    c.execute("INSERT INTO minimizers SELECT * FROM chr%s.minimizers" % chromosome)
    print("Detaching")
    c.execute('DETACH DATABASE chr%s' % (chromosome))

print("Commiting changes")
db.commit()




