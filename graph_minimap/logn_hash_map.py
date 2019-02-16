import numpy as np

class LogNHashMap:
    def __init__(self, sorted_hash_array):
        self._hashes = np.unique(sorted_hash_array)

    def hash(self, key):
        index = np.searchsorted(self._hashes, key)
        if self._hashes[index] != key:
            return None
        #assert self._hashes[index] == key
        return index

    def to_file(self, file_name):
        np.save(file_name, self._hashes)

    @classmethod
    def from_file(cls, file_name):
        data = np.load(file_name + ".npy")
        map = cls([])
        map._hashes = data
        return map

    def unhash(self, hash):
        return self._hashes[hash]


if __name__ == "__main__":
    hashes = np.array([4, 6, 6, 6, 6, 8, 8, 10])
    hashmap = LogNHashMap(hashes)

    print(hashmap.hash(8))
    print(hashmap.hash(6))

    size = 10000000
    hashes = np.sort(np.random.randint(0, 100000000000, size))
    map = LogNHashMap(hashes)
    print("Data created")

    for i, hash in enumerate(hashes):
        if i % 1000000 == 0:
            print(i)
        value = map.hash(hash)

    print("Done")

