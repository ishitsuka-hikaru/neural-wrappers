import numpy as np
import os
import sys
from scipy.ndimage import gaussian_filter

def standardizeData(data, mean, std):
	data -= mean
	data /= std
	return data

def minMaxNormalizeData(data, min, max):
	data -= min
	data /= (max - min)
	return data

def toCategorical(data, numClasses):
	numData = len(data)
	newData = np.zeros((numData, numClasses), dtype=np.uint8)
	newData[np.arange(numData), data] = 1
	return newData

# Labels can be None, in that case only data is available (testing cases without labels)
def makeGenerator(data, labels, batchSize):
	while True:
		numData = data.shape[0]
		numIterations = numData // batchSize + (numData % batchSize != 0)
		for i in range(numIterations):
			startIndex = i * batchSize
			endIndex = np.minimum((i + 1) * batchSize, numData)
			if not labels is None:
				yield data[startIndex : endIndex], labels[startIndex : endIndex]
			else:
				yield data[startIndex : endIndex]

def NoneAssert(conndition, noneCheck, message=""):
	if noneCheck:
		assert conndition, message

class LinePrinter:
	def __init__(self):
		self.maxLength = 0

	def print(self, message):
		if message[-1] == "\n":
			message = message[0 : -1]
			additional = "\n"
		else:
			additional = "\r"

		self.maxLength = np.maximum(len(message), self.maxLength)
		message += (self.maxLength - len(message)) * " " + additional
		sys.stdout.write(message)
		sys.stdout.flush()

# @brief Returns true if whatType is subclass of baseType. The parameters can be instantiated objects or types. In the
#  first case, the parameters are converted to their type and then the check is done.
def isBaseOf(whatType, baseType):
	if type(whatType) != type:
		whatType = type(whatType)
	if type(baseType) != type:
		baseType = type(baseType)
	return baseType in type(object).mro(whatType)

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
	return None if type(x) == type(None) else x if type(x) == list else [x]

# ["test"] and ["blah", "blah2"] => False
# ["blah2"] and ["blah", "blah2"] => True
def isSubsetOf(subset, set):
	for item in subset:
		if not item in set:
			return False
	return True

def changeDirectory(Dir, expectExist):
	assert os.path.exists(Dir) == expectExist
	print("Changing to working directory:", Dir)
	if expectExist == False:
		os.makedirs(Dir)
	os.chdir(Dir)

class RunningMean:
	def __init__(self, initValue=0):
		self.value = initValue
		self.count = 0

	def update(self, value, count):
		if not value is None:
			assert count > 0
			self.value += value
			self.count += count

	def get(self):
		return self.value / (self.count + 1e-5)

	def __repr__(self):
		return self.get()

	def __str__(self):
		return self.get()

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

def getGenerators(reader, miniBatchSize):
	generator = reader.iterate("train", miniBatchSize=miniBatchSize, maxPrefetch=1)
	numIters = reader.getNumIterations("train", miniBatchSize=miniBatchSize)
	valGenerator = reader.iterate("validation", miniBatchSize=miniBatchSize, maxPrefetch=1)
	valNumIters = reader.getNumIterations("validation", miniBatchSize=miniBatchSize)
	return generator, numIters, valGenerator, valNumIters

def tryReadImage(path, count=5, imgLib="opencv"):
	assert imgLib in ("opencv", "PIL")

	def readImageOpenCV(path):
		import cv2
		bgr_image = cv2.imread(path)
		b, g, r = cv2.split(bgr_image)
		image = cv2.merge([r, g, b]).astype(np.float32)
		return image

	def readImagePIL(path):
		from PIL import Image
		image = np.array(Image.open(path), dtype=np.float32)
		return image

	if imgLib == "opencv":
		f = readImageOpenCV
	elif imgLib == "PIL":
		f = readImagePIL

	i = 0
	while True:
		try:
			return f(path)
		except Exception as e:
			print(str(e))
			i += 1

			if i == count:
				raise Exception