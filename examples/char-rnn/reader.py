import numpy as np
from neural_wrappers.utilities import toCategorical
from neural_wrappers.readers import BatchedDatasetReader
from functools import partial

def f(d, i, sequenceSize):
	Items = []
	for _ in range(i.stop - i.start):
		startIx = np.random.randint(0, len(d) - sequenceSize - 1)
		item = d[startIx : startIx + sequenceSize + 1]
		Items.append(item)
	return Items

class Reader(BatchedDatasetReader):
	def __init__(self, path, sequenceSize, stepsPerEpoch):
		super().__init__(
			dataBuckets={"data" : ["sentence"]},
			dimGetter = {"sentence" : partial(f, sequenceSize=sequenceSize)},
			dimTransform = {}
		)
		self.datasetPath = path
		self.f = open(path, "r").read()
		chars = sorted(list(set(self.f)))
		self.charToIx = dict(zip(chars, range(len(chars))))
		self.ixToChar = dict(zip(range(len(chars)), chars))
		self.sequenceSize = sequenceSize
		self.stepsPerEpoch = stepsPerEpoch

	def sentenceToVector(self, sentence):
		res = []
		for c in sentence:
			v = self.charToIx[c]
			v = toCategorical([v], numClasses=len(self.charToIx))[0]
			res.append(v)
		res = np.array(res, dtype=np.float32)
		return res

	def vectorToSentence(self, vector):
		assert len(vector.shape) == 2, "TxOneHot. Shpae: %s" % str(vector.shape)
		res = ""
		for v in vector:
			c = np.argmax(v)
			c = self.ixToChar[c]
			res += c
		return res

	def sampleSentence(self, sequenceSize):
		startIx = np.random.randint(0, len(self.f) - sequenceSize - 1)
		Items = self.f[startIx : startIx + sequenceSize + 1]
		_X = Items[0 : -1]
		_t = Items[1 :]
		return _X, _t

	def getBatches(self):
		pass

	def getDataset(self):
		return self.f

	def getNumData(self):
		return self.stepsPerEpoch

	def __getitem__(self, index):
		item = super().__getitem__(index)
		item = item["data"]["sentence"]
		batchSize = len(item)

		X = np.zeros((self.sequenceSize, batchSize, len(self.charToIx)), dtype=np.float32)
		t = np.zeros((self.sequenceSize, batchSize, len(self.charToIx)), dtype=np.bool)

		for j in range(batchSize):
			sentence = item[j]
			_X, _t = sentence[0 : -1], sentence[1 : ]
			X[:, j] = self.sentenceToVector(_X)
			t[:, j] = self.sentenceToVector(_t)
		return (X, None), t
