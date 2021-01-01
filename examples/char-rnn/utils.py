import numpy as np
import torch as tr
from neural_wrappers.callbacks import Callback
from neural_wrappers.utilities import toCategorical
from neural_wrappers.pytorch import device

from reader import Reader

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
		# breakpoint()
		# charIndex = np.argmax(p)
		result += reader.ixToChar[charIndex]
		npOutput = toCategorical([charIndex], len(reader.charToIx)).astype(np.float32)[0]
		output = tr.from_numpy(npOutput).unsqueeze(dim=0).unsqueeze(dim=1).to(device)
	return seedText, result
