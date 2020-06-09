import h5py
import numpy as np
from .utils import smartIndexWrapper

def h5Print(data, level=0):
	if type(data) in (h5py._hl.files.File, h5py._hl.group.Group):
		for key in data:
			print("\n%s- %s" % ("  " * level, key), end="")
			h5Print(data[key], level=level+1)
	elif type(data) == h5py._hl.dataset.Dataset:
		print("Shape: %s. Type: %s" % (data.shape, data.dtype), end="")
	else:
		assert False, "Unexpected type %s" % (type(data))

def h5StoreDict(file, data):
	assert type(data) == dict
	for key in data:
		# If key is int, we need to convert it to Str, so we can store it in h5 file.
		sKey = str(key) if type(key) == int else key

		if type(data[key]) == dict:
			file.create_group(sKey)
			h5StoreDict(file[sKey], data[key])
		else:
			file[sKey] = data[key]

def h5ReadDict(data, N=None):
	if type(data) in (h5py._hl.files.File, h5py._hl.group.Group):
		res = {}
		for key in data:
			res[key] = h5ReadDict(data[key], N=N)
	elif type(data) == h5py._hl.dataset.Dataset:
		if N is None:
			res = data[()]
		elif type(N) is int:
			res = data[0 : N]
		elif type(N) in (list, np.ndarray):
			res = smartIndexWrapper(data, N)
	else:
		assert False, "Unexpected type %s" % (type(data))
	return res
