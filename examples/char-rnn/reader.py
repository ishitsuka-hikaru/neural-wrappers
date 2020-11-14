import numpy as np
from neural_wrappers.utilities import toCategorical

class Reader:
	def __init__(self, path, sequenceSize):
		self.f = open(path, "r").read()
		chars = sorted(list(set(self.f)))
		self.charToIx = dict(zip(chars, range(len(chars))))
		self.ixToChar = dict(zip(range(len(chars)), chars))
		self.sequenceSize = sequenceSize

	def sentenceToVector(self, sentence):
		res = []
		for c in sentence:
			v = self.charToIx[c]
			v = toCategorical([v], numClasses=len(self.charToIx))
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

	def iterate_once(self, numSteps, batchSize):
		for i in range(numSteps):
			X = np.zeros((self.sequenceSize, batchSize, len(self.charToIx)), dtype=np.float32)
			t = np.zeros((self.sequenceSize, batchSize, len(self.charToIx)), dtype=np.bool)

			for j in range(batchSize):
				_X, _t = self.sampleSentence(self.sequenceSize)
				X[:, j] = self.sentenceToVector(_X)
				t[:, j] = self.sentenceToVector(_t)
			yield (X, None), t
	
	def iterate(self, numSteps, batchSize):
		while True:
			generator = self.iterate_once(numSteps, batchSize)
			for item in generator:
				yield item