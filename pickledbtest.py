import pickledb

db = pickledb.load('example.db', False)

db.set("test", "test")

print(db.get("test"))

db.dump()

db.dump()
