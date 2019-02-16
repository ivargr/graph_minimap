import numpy as np
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def dict_to_file(d, file_name):
    logging.info("Writing dict to file %s. Now converting values" % file_name)
    values = np.array(list(d.values()))
    logging.info("Converting keys")
    keys = np.array(list(d.keys()))

    np.save(file_name + ".keys", values)
    np.save(file_name + ".values", keys)
    logging.info("Dict written to file")


def dict_from_file(file_name):
    logging.info("Reading data from files")
    values = np.load(file_name + ".values.npy")
    keys = np.load(file_name + ".keys.npy")

    logging.info("Converting to dict")

    d = dict(zip(keys, values))
    logging.info("Done reading from file")
    return d


if __name__ == "__main__":
    import random
    dict_size = 100000000

    logging.info("Creating data")
    d = dict(zip(np.random.randint(0, 100000000000, dict_size), np.random.randint(0, 100000000000, dict_size)))

    logging.info("Data created")
    dict_to_file(d, "testdict")
    new_d = dict_from_file("testdict")
    print(len(new_d.keys()))

