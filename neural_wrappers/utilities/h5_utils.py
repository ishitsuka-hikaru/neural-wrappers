import h5py
import numpy as np

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
	if type(data) == dict:
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
			res = h5ReadSmartIndexing(data, N)
	else:
		assert False, "Unexpected type %s" % (type(data))
	return res

def h5ReadSmartIndexing(data, indexes):
	# Flatten the indexes [[1, 3], [15, 13]] => [1, 3, 15, 13]
	indexes = np.array(indexes, dtype=np.uint32)
	flattenedIndexes = indexes.flatten()
	N = len(flattenedIndexes)

	flattenedShape = (N, *data.shape[1 : ])
	finalShape = (*indexes.shape, *data.shape[1 : ])
	# Retrieve the items 1 by 1 from the flattened version of the indexes
	result = np.zeros(flattenedShape, data.dtype)
	for i in range(N):
		result[i] = data[flattenedIndexes[i]]
	# Reshape the data accordingly
	result = result.reshape(finalShape)
	return result
