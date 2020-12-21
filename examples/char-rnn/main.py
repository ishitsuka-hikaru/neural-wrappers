# Char-rnn like the one from: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
# Download input txt frile from that blog (i.e. Shakespeare)
import numpy as np
import torch as tr
import torch.optim as optim
import time
from argparse import ArgumentParser

from neural_wrappers.pytorch import device, FeedForwardNetwork
from neural_wrappers.readers import DatasetReader, StaticBatchedDatasetReader
from neural_wrappers.callbacks import SaveModels, Callback
from neural_wrappers.utilities import toCategorical

from reader import Reader
from model import Model

class SampleCallback(Callback):
	def __init__(self, reader):
		super().__init__()
		self.reader = reader

	def onEpochEnd(self, **kwargs):
		seed, result = sample(kwargs["model"], self.reader, numIters=200)
		print("Seed: %s" % seed)
		print("Result: %s" % result)
		print("\n___________________________________________________________________\n")

	def onCallbackSave(self, **kwargs):
		state = self.reader.datasetPath, self.reader.sequenceSize, self.reader.stepsPerEpoch
		self.reader = None
		return state

	def onCallbackLoad(self, additional, **kwargs):
		datasetPath, sequenceSize, stepsPerEpoch = additional
		self.reader = Reader(datasetPath, sequenceSize, stepsPerEpoch)

def lossFn(output, target):
	# Batched binary cross entropy
	output, hidden = output
	return -tr.log(output[target] + 1e-5).mean()

def sample(model, reader, numIters, seedText=None):
	if seedText is None:
		seedText = reader.sampleSentence(sequenceSize=reader.sequenceSize)[0]
	tensorSeed = np.expand_dims(reader.sentenceToVector(seedText), axis=0)

	hprev = None
	for i in range(len(seedText)):
		input = tensorSeed[:, i : i + 1]
		output = tr.from_numpy(input).to(device)
		_, hprev = model.forward([output, hprev])

	result = ""
	for i in range(200):
		output, hprev = model.forward([output, hprev])
		p = output.detach().to("cpu").numpy()[0].flatten()
		charIndex = np.random.choice(range(len(reader.charToIx)), p=p)
		result += reader.ixToChar[charIndex]
		npOutput = toCategorical(charIndex, len(reader.charToIx)).astype(np.float32)
		output = tr.from_numpy(npOutput).unsqueeze(dim=0).unsqueeze(dim=1).to(device)
	return seedText, result

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("type")
	parser.add_argument("datasetPath")
	parser.add_argument("--weightsFile")
	parser.add_argument("--batchSize", type=int, default=5)
	parser.add_argument("--numEpochs", type=int, default=100)
	parser.add_argument("--stepsPerEpoch", type=int, default=10000)
	parser.add_argument("--sequenceSize", type=int, default=10)
	parser.add_argument("--seedText")
	parser.add_argument("--cellType", default="LSTM")
	args = parser.parse_args()
	return args

def main():
	args = getArgs()
	reader = StaticBatchedDatasetReader(Reader(args.datasetPath, args.sequenceSize, args.stepsPerEpoch), args.batchSize)
	generator = reader.iterateForever()

	model = Model(cellType=args.cellType, inputSize=len(reader.charToIx), hiddenSize=100).to(device)
	model.setOptimizer(optim.SGD, lr=0.001, momentum=0.9)
	model.setCriterion(lossFn)
	model.addCallbacks([SaveModels(mode="last", metricName="Loss"), SampleCallback(reader)])
	print(model.summary())

	if args.type == "train":
		model.train_generator(generator, stepsPerEpoch=len(generator), numEpochs=args.numEpochs)
	elif args.type == "test":
		model.loadModel(sys.argv[3])

		while True:
			result = sample(model, reader, numIters=200, seedText=args.seedText)
			print(result + "\n___________________________________________________________________")
			time.sleep(1)

if __name__ == "__main__":
	main()