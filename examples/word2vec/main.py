import numpy as np
import torch as tr
import sys
import torch.optim as optim
from argparse import ArgumentParser

from neural_wrappers.models import ModelWord2Vec
from neural_wrappers.readers import Word2VecReader
from neural_wrappers.readers.word2vec_reader import Reader
from neural_wrappers.callbacks import PlotMetricsCallback, Callback
from neural_wrappers.pytorch import NeuralNetworkPyTorch

device = tr.device("cuda") if tr.cuda.is_available() else tr.device("cpu")

class SaveEmbeddingCallback(Callback):
	def onEpochEnd(self, **kwargs):
		kwargs["model"].saveEmbeddings(outFile="emb-epoch%d.vec" % (kwargs["epoch"]), quiet=True)

def getArgs():
	parser = ArgumentParser()
	parser.add_argument("corpus_path")

	parser.add_argument("--embedding_size", default=5, type=int)
	parser.add_argument("--window_size", default=2, type=int)
	parser.add_argument("--num_negative_samples", default=4, type=int)
	parser.add_argument("--batch_size", default=10, type=int)
	parser.add_argument("--num_epochs", default=100, type=int)

	args = parser.parse_args()
	return args

# Model without negative sampling (simple autoencoder)
class Model(ModelWord2Vec):
	def forward(self, x):
		wordEmb = self.embIn(x)
		y2 = tr.mm(wordEmb, self.embOut.weight.t())
		y3 = tr.softmax(y2, dim=-1)
		return y3

	def networkAlgorithm(self, trInputs, trLabels):
		trResults = self.forward(trInputs)
		trLoss = self.criterion(trResults, trLabels)
		return trResults, trLoss

def train(tokenizedCorpus, embeddingSize=5, windowSize=2, numNegative=4, miniBatchSize=2, numEpochs=150):
	reader = Word2VecReader(tokenizedCorpus, windowSize=windowSize, numNegative=numNegative)
	model = ModelWord2Vec(reader.dictionary, embeddingSize=embeddingSize).to(device)

	# reader = Reader(tokenizedCorpus, windowSize=windowSize)
	# model = Model(reader.dictionary, embeddingSize=embeddingSize).to(device)

	generator = reader.iterate(miniBatchSize=miniBatchSize)
	numIters = reader.getNumIterations()

	model.setOptimizer(optim.SGD, lr=0.01, momentum=0.5)
	plotCallback = PlotMetricsCallback(["Loss"], ["min"])
	model.addCallbacks([plotCallback, SaveEmbeddingCallback()])
	print(model.summary())

	model.train_generator(generator, numIters, numEpochs=numEpochs)
	return model

def main():
	args = getArgs()
	corpus = list(map(lambda x : x.strip("\n"), open(args.corpus_path, "r").readlines()))
	model = train(corpus, embeddingSize=args.embedding_size, windowSize=args.window_size, \
		numNegative=args.num_negative_samples, miniBatchSize=args.batch_size, numEpochs=args.num_epochs)

if __name__ == "__main__":
	main()
