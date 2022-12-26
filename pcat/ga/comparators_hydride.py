import numpy as np


class DataComparator:
    """Compares the calculated hash strings. These strings should be stored
       in atoms.info['data'][key1] and
       atoms.info['data'][key2] ...
       where the keys should be supplied as parameters i.e.
       StringComparator(key1, key2, ...)
    """

    def __init__(self, *keys):
        self.keys = keys

    def looks_like(self, a1, a2):
        for k in self.keys:
            if a1.info['data'][k] == a2.info['data'][k]:
                return True
        return False

