# Char-rnn like the one from: http://karpathy.github.io/2015/05/21/rnn-effectiveness/
# Download input txt frile from that blog (i.e. Shakespeare)
import sys
import time
import numpy as np
import torch as tr
import torch.optim as optim
from neural_wrappers.pytorch import device
from neural_wrappers.callbacks import SaveModels, Callback

from reader import Reader
from model import Model

def lossFn(output, target):
	# Batched binary cross entropy
	output = output[0]
	L = -tr.log(output[target] + 1e-5).mean()
	return L

class SampleCallback(Callback):
	def __init__(self, reader, N):
		super().__init__()
		self.reader = reader
		self.N = N

	def onEpochEnd(self, **kwargs):
		model = kwargs["model"]
		inputSentence = self.reader.sampleSentence(self.reader.sequenceSize)[0]

		# v = tr.from_numpy(self.reader.sentenceToVector(inputSentence)).unsqueeze(dim=1).to(device)
		v = np.expand_dims(self.reader.sentenceToVector(inputSentence), axis=1)
		Str = ""
		out, h = model.npForward([v, None])
		charOut = self.reader.vectorToSentence(out[-1])
		v = np.expand_dims(self.reader.sentenceToVector(charOut), axis=1)

		print("Input sentence: %s" % inputSentence)
		for i in range(self.N):
			with tr.no_grad():
				out, h = model.npForward([v, h])
			charOut = self.reader.vectorToSentence(out[-1])
			v = np.expand_dims(self.reader.sentenceToVector(charOut), axis=1)

			Str += charOut
			# print("Input: %s | Predicted: %s (%2.2f)" % (Str, charOut, out.max() ))
		print("Predicted: %s" % Str)

# def sample(model, seedText, numIters, hprev=None):
# 	tensorSeed = TextUtils.toCategorical(np.expand_dims(TextUtils.textToInt(seedText), axis=0))

# 	hprev = None
# 	for i in range(len(seedText)):
# 		output = tr.from_numpy(tensorSeed[:, i]).to(device)
# 		_, hprev = model.forward(output, hprev)

# 	result = ""
# 	for i in range(200):
# 		output, hprev = model.forward(output, hprev)
# 		p = output.detach().to("cpu").numpy()[0].flatten()
# 		charIndex = np.random.choice(range(len(TextUtils.alphabet)), p=p)
# 		result += TextUtils.alphabet[charIndex]
# 		output = tr.from_numpy(np.expand_dims(TextUtils.toCategorical(charIndex), axis=0)).to(device)
# 	return result

def main():
	assert sys.argv[1] in ("train", "test")
	sequenceSize = 10
	reader = Reader(sys.argv[2], sequenceSize=sequenceSize)
	I, H, O = len(reader.charToIx), 30, len(reader.charToIx)
	model = Model(cellType="LSTM", inputSize=I, hiddenSize=H, outputSize=O).to(device)
	model.addCallbacks([SaveModels("best", "Loss"), SampleCallback(reader, N=50)])
	print(model.summary())

	if sys.argv[1] == "train":
		numSteps = 10000
		generator = reader.iterate(numSteps=numSteps, batchSize=2)

		model.setOptimizer(optim.SGD, lr=0.001, momentum=0.9)
		model.setCriterion(lossFn)
		model.train_generator(generator, stepsPerEpoch=numSteps, numEpochs=100)

	elif sys.argv[1] == "test":
		model.load_weights(sys.argv[2])

		while True:
			result = sample(model, seedText="Hello", numIters=200)
			print(result + "\n___________________________________________________________________")
			time.sleep(1)

if __name__ == "__main__":
	main()