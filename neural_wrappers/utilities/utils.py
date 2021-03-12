import numpy as np
import os
import pickle
from collections import OrderedDict
from typing import Dict, Sequence, Union, Iterable, List, Optional
from functools import reduce
from pathlib import Path
from .np_utils import npCloseEnough
from .type_utils import NWNumber, NWSequence, NWDict, isBaseOf, T
from .debug import Debug

def NoneAssert(conndition, noneCheck, message=""):
	if noneCheck:
		assert conndition, message

# Stubs for identity functions, first is used for 1 parameter f(x) = x, second is used for more than one parameter,
#  such as f(x, y, z) = (x, y, z)
def identity(x, **kwargs):
	return x

def identityVar(*args):
	return args

# Stub for making a list, used by various code parts, where the user may provide a single element for a use-case where
#  he'd have to use a 1-element list. This handles that case, so the overall API uses lists, but user provides
#  just an element. If None, just return None.
def makeList(x):
	return None if type(x) == type(None) else list(x) if type(x) in (list, set, tuple) else [x]

# ["test"] and ["blah", "blah2"] => False
# ["blah2"] and ["blah", "blah2"] => True
def isSubsetOf(subset, set):
	for item in subset:
		if not item in set:
			return False
	return True

def changeDirectory(Dir:str, expectExist:Optional[bool]=None):
	assert not Dir is None, ""
	if expectExist in (True, False):
		assert os.path.exists(Dir) == expectExist, "Exists: %s" % Dir
	Debug.print("Changing to working directory: %s" % Dir)
	if expectExist == False or (expectExist == None and not os.path.isdir(Dir)):
		os.makedirs(Dir)
	os.chdir(Dir)

# Given a graph as a dict {Node : [Dependencies]}, returns a list [Node] ordered with a correct topological sort order
# Applies Kahn's algorithm: https://ocw.cs.pub.ro/courses/pa/laboratoare/laborator-07
def topologicalSort(depGraph):
	L, S = [], []

	# First step is to create a regular graph of {Node : [Children]}
	graph = {k : [] for k in depGraph.keys()}
	for key in depGraph:
		for parent in depGraph[key]:
			graph[parent].append(key)
		# Transform the depGraph into a list of number of in-nodes
		depGraph[key] = len(depGraph[key])
		# Add nodes with no dependencies and start BFS from them
		if depGraph[key] == 0:
			S.append(key)

	while len(S) > 0:
		u = S.pop()
		L.append(u)

		for v in graph[u]:
			depGraph[v] -= 1
			if depGraph[v] == 0:
				S.append(v)

	for key in depGraph:
		if depGraph[key] != 0:
			raise Exception("Graph is not acyclical")
	return L

# @brief Utility function that returns a generator and the number of iterations for that generator.
#  Supports multiple keys.
# @param[in] reader A DatasetReader object for the used dataset
# @param[in] maxPrefetch Whether to use prefetch_generator library to use multiple threads to read N iterations ahead.
# @param[in] keys The keys used to return pairs of (generator, iterations). Defaults to "train", "validation"
# @return A flattened list of pairs of type (generator, iteraions). For the values, we get 4 items.
def getGenerators(reader, batchSize:int=None, maxPrefetch:int=1):
	if not batchSize is None:
		assert hasattr(reader, "setBatchSize"), "reader has no method setBatchSizes. Call getGenerators with None."
		reader.setBatchSize(batchSize)
	# breakpoint()
	generator = reader.iterateForever(maxPrefetch=maxPrefetch)
	return generator, len(generator)

# Deep check if two items are equal. Dicts are checked value by value and numpy array are compared using "closeEnough"
#  method
def deepCheckEqual(a, b):
	if type(a) != type(b):
		Debug.print("Types %s and %s differ." % (type(a), type(b)))
		return False
	Type = type(a)
	if Type in (dict, OrderedDict):
		for key in a:
			if not deepCheckEqual(a[key], b[key]):
				return False
		return True
	elif Type == np.ndarray:
		if not a.shape == b.shape:
			return False
		return npCloseEnough(a, b)
	elif Type in (list, tuple):
		if not len(a) == len(b):
			return False
		for i in range(len(a)):
			if not deepCheckEqual(a[i], b[i]):
				return False
		return True
	else:
		return a == b
	assert False, "Shouldn't reach here"
	return False

def isPicklable(item):
	try:
		_ = pickle.dumps(item)
		return True
	except Exception as e:
		Debug.print("Item is not pickable: %s" % (e))
		return False

# Flatten the indexes [[1, 3], [15, 13]] => [1, 3, 15, 13] and then calls f(data, 1), f(data, 3), ..., step by step
def smartIndexWrapper(data, indexes, f = lambda data, index : data[index]):
	# Flatten the indexes [[1, 3], [15, 13]] => [1, 3, 15, 13]
	indexes = np.array(indexes, dtype=np.uint32)
	flattenedIndexes = indexes.flatten()
	N = len(flattenedIndexes)
	assert N > 0

	result = []
	for i in range(N):
		result.append(f(data, flattenedIndexes[i]))
	finalShape = (*indexes.shape, *result[0].shape)
	result = np.array(result).reshape(finalShape)
	return result

def getFormattedStr(item : Union[np.ndarray, NWNumber, NWSequence, NWDict], precision : int) -> str: \
	# type: ignore
	formatStr = "%%2.%df" % (precision)
	if type(item) in NWNumber.__args__: # type: ignore
		return formatStr % (item) # type: ignore
	elif type(item) in NWSequence.__args__: # type: ignore
		return [formatStr % (x) for x in item] # type: ignore
	elif type(item) in NWDict.__args__: # type: ignore
		return {k : formatStr % (item[k]) for k in item} # type: ignore
	elif isinstance(item, (np.int32, np.float32, np.float64)):
		return formatStr % (item)
	assert False, "Unknown type: %s" % (type(item))

def flattenList(x : Iterable[List[T]]) -> List[T]:
	return reduce(lambda a, b : a + b, x)

def tryStr(x) -> str:
	try:
		return str(x, "utf8")
	except Exception:
		return x

# @brief Return the value of a nested dictionary key
# @param[in] d The potentially nested dictionary
# @param[in] k The potentially nested lookup key
# @return The value of the potentially nested key
def deepDictGet(d:Dict, k):
	if isinstance(k, (tuple, list)):
		if len(k) == 1:
			return d[k]
		else:
			return deepDictGet(d[k[0]], k[1 :])
	else:
		return d[k]

# Given a path to a file/directory, return the absolute real path.
def fullPath(x:str):
	return Path(os.path.abspath(os.path.realpath(x)))