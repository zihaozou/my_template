import torch

torch.multiprocessing.set_sharing_strategy("file_system")
import torch.utils.data as data
import lmdb
import pickle
from functools import reduce
from pathlib import Path
import multiprocessing as mp

"""
The utils.py file contains a Python class called LMDBDatabase that provides a simple interface for working with LMDB databases. The class provides methods for getting, setting, and deleting key-value pairs, as well as iterating over the values in the database. The class also includes methods for getting the keys in the database and getting the length of the database.

The setup decorator in the utils.py file ensures that the LMDBDatabase class works correctly under a multiprocessing environment. This is important because LMDB is not thread-safe.
"""


def setup(func):
    def wrapper(ref, *args, **kwargs):
        if ref.pid != ref.base_pid or ref.env is None:
            ref.env = lmdb.open(
                str(ref.path),
                readonly=ref.readonly,
                lock=False,
                readahead=False,
                meminit=False,
                subdir=True,
                map_size=1099511627776 * 2,
                map_async=True,
            )
            ref.base_pid = ref.pid
        return func(ref, *args, **kwargs)

    return wrapper


class LMDBDatabase:
    def __init__(self, path, readonly=True):
        self.path = Path(path)
        if not self.path.exists():
            self.path.mkdir(parents=True)
        self.readonly = readonly
        self.base_pid = self.pid
        self.env = None

    @property
    def pid(self):
        return mp.current_process().pid

    @setup
    def keys(self):
        with self.env.begin() as txn:
            return list(map(lambda x: x.decode(), txn.cursor().iternext(values=False)))

    @setup
    def __len__(self):
        with self.env.begin() as txn:
            return txn.stat()["entries"]

    @setup
    def __getitem__(self, key):
        if isinstance(key, int):
            if key >= len(self):
                raise KeyError(f"Key {key} not found")
            key = self.keys()[key]
            return self[key]
        if key not in self.keys():
            raise KeyError(f"Key {key} not found")
        with self.env.begin() as txn:
            return pickle.loads(txn.get(key.encode("ascii")))

    @setup
    def __setitem__(self, key, value):
        if self.readonly:
            raise RuntimeError("Cannot set item on read-only database")
        if isinstance(key, int):
            if key >= len(self):
                raise KeyError(f"Key {key} not found")
            key = self.keys()[key]
            self[key] = value
            return
        with self.env.begin(write=True) as txn:
            txn.put(key.encode("ascii"), pickle.dumps(value), overwrite=True)

    @setup
    def __delitem__(self, key):
        if self.readonly:
            raise RuntimeError("Cannot delete item on read-only database")
        if isinstance(key, int):
            if key >= len(self):
                raise KeyError(f"Key {key} not found")
            key = self.keys()[key]
            del self[key]
            return
        if key not in self.keys():
            raise KeyError(f"Key {key} not found")
        with self.env.begin(write=True) as txn:
            txn.delete(key.encode("ascii"))

    def __del__(self):
        if self.env is not None:
            if not self.readonly:
                self.env.sync()
            self.env.close()

    def __iter__(self):
        for key in self.keys():
            yield self[key]

    def __contains__(self, key):
        return key in self.keys()

    def __repr__(self):
        return f"{self.__class__.__name__}({self.path}, {len(self)})"


class MetaHandler(type):
    def __new__(cls, name, bases, attrs):
        new_class = type.__new__(cls, name, bases, attrs)
        super_class = super(new_class, new_class)
        if hasattr(super_class, "__getitem_wrap__") and hasattr(
            new_class, "__getitem__"
        ):
            super_getitem_wrap = getattr(super_class, "__getitem_wrap__")
            sub_getitem = getattr(new_class, "__getitem__")

            def __new_getitem__(self, *args, **kwargs):
                return super_getitem_wrap(self, sub_getitem, *args, **kwargs)

            setattr(new_class, "__getitem__", __new_getitem__)
        return new_class


class CacheDataset(data.Dataset, metaclass=MetaHandler):
    def __init__(self):
        super().__init__()
        self.manager = mp.Manager()
        self.cache = self.manager.dict()

    def __getitem_wrap__(self, func, idx):
        data = self.cache.get(idx, func(self, idx))
        self.cache[idx] = data
        return data


import unittest
import tempfile
import shutil


class TestLMDBDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db = LMDBDatabase(self.temp_dir, readonly=False)
        self.db["key1"] = "value1"
        self.db["key2"] = "value2"

    def tearDown(self):
        del self.db
        shutil.rmtree(self.temp_dir)

    def test_getitem(self):
        self.assertEqual(self.db["key1"], "value1")
        self.assertEqual(self.db["key2"], "value2")
        with self.assertRaises(KeyError):
            self.db["key3"]

    def test_setitem(self):
        self.db["key1"] = "new_value1"
        self.assertEqual(self.db["key1"], "new_value1")
        with self.assertRaises(RuntimeError):
            self.db.readonly = True
            self.db["key3"] = "value3"

    def test_delitem(self):
        del self.db["key1"]
        with self.assertRaises(KeyError):
            self.db["key1"]
        with self.assertRaises(RuntimeError):
            self.db.readonly = True
            del self.db["key2"]

    def test_len(self):
        self.assertEqual(len(self.db), 2)

    def test_iter(self):
        self.assertEqual(list(iter(self.db)), ["value1", "value2"])

    def test_contains(self):
        self.assertTrue("key1" in self.db)
        self.assertFalse("key3" in self.db)


if __name__ == "__main__":
    unittest.main()
