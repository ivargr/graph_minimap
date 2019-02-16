from random import randint
import judy
import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
import pickledb
import pickle
from preshed.maps import PreshMap
from sklearn.utils.fast_dict import IntFloatDict

def test():
    logging.info("Starting")
    a = {}
    a = PreshMap()
    #a = judy.JudyIntObjectMap()
    #db = pickledb.load("pickledb.db", False)

    dict_size = 100000000

    keys = np.random.randint(0, 1000000000000, dict_size)
    values = np.random.randint(0, 1000000000, dict_size).astype(np.float64)
    logging.info("Creating")
    a = IntFloatDict(keys, values)
    logging.info("Done")
    return

    #a = dict(zip(np.random.randint(0, 100000000000, dict_size), np.random.randint(0, 100000000000, dict_size)))
    for i in range(0, 10000):
        if i % 1000000 == 0:
            print(i)

        number = randint(0, 10000000000000)

        a[number] = randint(0, 3000000000)
        #db.set(str(number), randint(0, 3000000000))

    #db.dump()
    logging.info("writing to file")
    with open("testdict.pckl", "wb") as f:
        pickle.dump(a, f, protocol=4)
    logging.info("Wrote to file")


test()
b = input("Stopp")

    
