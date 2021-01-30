# Char-rnn like the one from: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
# Download input txt frile from that blog (i.e. Shakespeare)
import numpy as np
import torch as tr
import torch.optim as optim
import time
from argparse import ArgumentParser

from neural_wrappers.pytorch import device, FeedForwardNetwork
from neural_wrappers.readers import DatasetReader, StaticBatchedDatasetReader
from neural_wrappers.callbacks import SaveModels, PlotMetrics

from reader import Reader
from model import Model
from utils import SampleCallback, sample

def lossFn(output, target):
	# Batched binary cross entropy
	output, hidden = output
	return -tr.log(output[target] + 1e-5).mean()

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
	model.setOptimizer(optim.Adam, lr=0.01)
	model.setCriterion(lossFn)
	model.addCallbacks([SaveModels(mode="last", metricName="Loss"), SampleCallback(reader), PlotMetrics(["Loss"])])
	print(model.summary())

	if args.type == "train":
		model.trainGenerator(generator, numEpochs=args.numEpochs)
	elif args.type == "test":
		model.loadModel(sys.argv[3])

		while True:
			result = sample(model, reader, numIters=200, seedText=args.seedText)
			print(result + "\n___________________________________________________________________")
			time.sleep(1)

if __name__ == "__main__":
	main()