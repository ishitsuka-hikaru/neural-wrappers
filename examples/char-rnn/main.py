# Char-rnn like the one from: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
# Download input txt frile from that blog (i.e. Shakespeare)
import numpy as np
import torch as tr
import torch.nn as nn
import torch.optim as optim
import sys
import time
from neural_wrappers.pytorch import RecurrentNeuralNetworkPyTorch, maybeCuda, maybeCpu
from neural_wrappers.readers import DatasetReader
from neural_wrappers.callbacks import SaveModels
from torch.autograd import Variable

class RNN_PyTorch(RecurrentNeuralNetworkPyTorch):
	def __init__(self, cellType, hiddenSize):
		super(RNN_PyTorch, self).__init__()
		assert cellType in ("RNN", "GRU", "LSTM")
		self.hiddenSize = hiddenSize
		self.inputSize = len(TextUtils.alphabet)
		self.outputSize = len(TextUtils.alphabet)
		self.cellType = cellType

		if self.cellType == "RNN":
			self.rnn = nn.RNN(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=1)
		elif self.cellType == "GRU":
			self.rnn = nn.GRU(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=1)
		elif self.cellType == "LSTM":
			self.rnn = nn.GRU(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=1)
		self.fc1 = nn.Linear(self.hiddenSize, self.outputSize)

	def forward(self, input, hidden):
		miniBatch = input.shape[0]
		# MB x InputShape => MB x 1 x InputShape (because we go steps 1 by 1)
		input = input.view(input.shape[0], 1, input.shape[1]).contiguous()
		output, hidden = self.rnn(input, hidden)
		output = self.fc1(output)
		output = nn.functional.softmax(output, dim=2)
		output = output.view(miniBatch, self.outputSize)
		return output, hidden

class TextUtils:
	alphabet = ['\n', ' ', '!', '$', '&', "'", ',', '-', '.', '3', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', \
		'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', \
		'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', \
		'z']

	@staticmethod
	def textToInt(text):
		return np.array(list(map(lambda x : TextUtils.alphabet.index(x), text)))

	@staticmethod
	def intToText(intSequence):
		return "".join(map(lambda x : TextUtils.alphabet[x], intSequence))

	@staticmethod
	def toCategorical(input):
		numClasses = len(TextUtils.alphabet)
		# int => (M, )
		if type(input) in (int, np.int32, np.int64):
			x = tr.zeros(numClasses,)
			x = np.zeros((numClasses, ), dtype=np.float32)
			x[input] = 1
		else:
			# (MB, N, ) => (MB, N, M)
			mbSize = input.shape[0]
			seqSize = input.shape[1]
			assert np.max(input) < numClasses
			x = np.zeros((mbSize, seqSize, numClasses), dtype=np.float32)
			for i in range(mbSize):
				x[i, np.arange(seqSize), input[i]] = 1
		return x

class Dataset(DatasetReader):
	def __init__(self, textData):
		self.textData = textData

	def __str__(self):
		return "[Dataset] Data has %d characters, with an alphabet of %d characters." % (len(self.textData), \
			len(TextUtils.alphabet))

	def iterate_once(self, type, sequenceSize):
		numData = len(self.textData)
		numSequences = self.getNumIterations(sequenceSize)

		for j in range(numSequences):
			seqStartIndex, seqEndIndex = j * sequenceSize, np.minimum((j + 1) * sequenceSize, (numData - 1))
			inputs = np.expand_dims(TextUtils.textToInt(self.textData[seqStartIndex : seqEndIndex]), axis=0)
			targets = np.expand_dims(TextUtils.textToInt(self.textData[seqStartIndex + 1 : seqEndIndex + 1]), axis=0)
			yield TextUtils.toCategorical(inputs), targets

	# All formulas are with numData - 1, because the target is the sequence shifted by 1 character to right
	def getNumIterations(self, sequenceSize):
		numData = len(self.textData)
		return (numData - 1) // sequenceSize + ((numData - 1) % sequenceSize != 0)

def lossFn(output, target):
	# Batched binary cross entropy
	return tr.mean(-tr.log(output[:, target] + 1e-5))

def sample(model, seedText, numIters, hprev=None):
	tensorSeed = TextUtils.toCategorical(np.expand_dims(TextUtils.textToInt(seedText), axis=0))

	hprev = None
	for i in range(len(seedText)):
		output = maybeCuda(Variable(tr.from_numpy(tensorSeed[:, i])))
		_, hprev = model.forward(output, hprev)

	result = ""
	for i in range(200):
		output, hprev = model.forward(output, hprev)
		p = maybeCpu(output.data).numpy()[0].flatten()
		charIndex = np.random.choice(range(len(TextUtils.alphabet)), p=p)
		result += TextUtils.alphabet[charIndex]
		output = maybeCuda(Variable(tr.from_numpy(np.expand_dims(TextUtils.toCategorical(charIndex), axis=0))))
	return result

def main():
	assert sys.argv[1] in ("train", "test")
	sequenceSize = 5
	model = maybeCuda(RNN_PyTorch(cellType="LSTM", hiddenSize=100))
	print(model.summary())

	if sys.argv[1] == "train":
		dataset = Dataset(open(sys.argv[2], "r").read())
		print(dataset)
		generator = dataset.iterate(None, miniBatchSize=sequenceSize)
		numIterations = dataset.getNumIterations(sequenceSize=sequenceSize)

		model.setOptimizer(optim.SGD, lr=0.001, momentum=0.9)
		model.setCriterion(lossFn)
		model.train_generator(generator, stepsPerEpoch=numIterations, numEpochs=10, callbacks=[SaveModels(type="all")])

	elif sys.argv[1] == "test":
		model.load_weights(sys.argv[2])

		while True:
			result = sample(model, seedText="Hello", numIters=200)
			print(result + "\n___________________________________________________________________")
			time.sleep(1)

if __name__ == "__main__":
	main()