from .network import NeuralNetworkPyTorch
from .utils import maybeCuda, maybeCpu
from neural_wrappers.utilities import LinePrinter
from datetime import datetime

import torch as tr
from torch.autograd import Variable

class GenerativeAdversialNetwork(NeuralNetworkPyTorch):
	def __init__(self, generator, discriminator):
		super().__init__()
		self.generator = generator
		self.discriminator = discriminator
		self.linePrinter = LinePrinter()

	def run_one_epoch(self, generator, stepsPerEpoch, generatorSteps, discriminatorSteps, \
		callbacks=[], optimize=False, printMessage=False):
		assert callbacks == []

		npLossD, npLossG = 0, 0
		startTime = datetime.now()

		for i in range(stepsPerEpoch):
			images, _ = next(generator)
			MB = images.shape[0]

			fakeLabels = Variable(maybeCuda(tr.zeros(MB)), requires_grad=False)
			realLabels = Variable(maybeCuda(tr.ones(MB)), requires_grad=False)

			# lossG = log(1 - D(G(z)))
			for step in range(generatorSteps):
				# Train generator
				randomInputsG = Variable(maybeCuda(tr.randn(MB, 100)), requires_grad=False)

				outG = self.generator.forward(randomInputsG)
				outDG = self.discriminator.forward(outG)
				lossG = self.criterion(outDG, realLabels)
				currentLossG = maybeCpu(lossG.data).numpy() / generatorSteps
				npLossG += currentLossG

				if optimize:
					self.generator.optimizer.zero_grad()
					lossG.backward()
					self.generator.optimizer.step()

			for step in range(discriminatorSteps):
				# Train discriminator, by using half images and half random noise
				images = Variable(maybeCuda(tr.from_numpy(images)), requires_grad=False)
				randomInputsG = Variable(maybeCuda(tr.randn(MB, 100)), requires_grad=False)

				outDReal = self.discriminator.forward(images)
				outDFake = self.discriminator.forward(self.generator.forward(randomInputsG))

				# lossD = log(G(x)) + log(1 - D(G(z)))
				lossD = self.criterion(outDReal, realLabels) + self.criterion(outDFake, fakeLabels)
				currentLossD = maybeCpu(lossD.data).numpy() / discriminatorSteps
				npLossD += currentLossD

				if optimize:
					self.discriminator.optimizer.zero_grad()
					lossD.backward()
					self.discriminator.optimizer.step()
	
			iterFinishTime = (datetime.now() - startTime)
			ETA = abs(iterFinishTime / (i + 1) * stepsPerEpoch - iterFinishTime)
			if printMessage:
				self.linePrinter.print("Epoch: %d. Iteration: %d/%d Loss D: %2.2f. Loss G: %2.2f. ETA: %s" % \
					(self.currentEpoch, i, stepsPerEpoch, npLossD / (i + 1), npLossG / (i + 1), ETA))
		return npLossD / stepsPerEpoch, npLossG / stepsPerEpoch

	def train_generator(self, generator, stepsPerEpoch, numEpochs, generatorSteps=1, discriminatorSteps=1):
		assert generatorSteps >= 1 and discriminatorSteps >= 1
		self.linePrinter.print("Training for %d epochs...\n" % (numEpochs))

		while self.currentEpoch < numEpochs + 1:
			self.trainHistory.append({})
			# self.currentEpoch = epoch + 1
			npLossD, npLossG = self.run_one_epoch(generator, stepsPerEpoch, generatorSteps=generatorSteps, \
				discriminatorSteps=discriminatorSteps, callbacks=[], optimize=True, printMessage=True)
			self.linePrinter.print("Epoch: %d/%d. Loss D: %2.2f. Loss G: %2.2f\n" % \
				(self.currentEpoch, numEpochs, npLossD, npLossG))

			self.trainHistory[-1] = {
				"generatorLoss" : npLossG,
				"discriminatorLoss" : npLossD
			}

			self.save_model("GAN.pkl")
			self.currentEpoch += 1

	def cuda(self):
		self.generator = maybeCuda(self.generator)
		self.discriminator = maybeCuda(self.discriminator)
		return self

	def save_model(self, path):
		generatorState = {
			"weights" : list(map(lambda x : x.cpu(), self.generator.parameters())),
			"optimizer_type" : type(self.generator.optimizer),
			"optimizer_state" : self.generator.optimizer.state_dict(),
		}

		discriminatorState = {
			"weights" : list(map(lambda x : x.cpu(), self.discriminator.parameters())),
			"optimizer_type" : type(self.discriminator.optimizer),
			"optimizer_state" : self.discriminator.optimizer.state_dict(),
		}

		state = {
			"generatorState" : generatorState,
			"discriminatorState" : discriminatorState,
			"history_dict" : self.trainHistory
		}

		tr.save(state, path)

	def load_model(self, path):
		loaded_model = tr.load(path)
		generatorState = loaded_model["generatorState"]
		discriminatorState = loaded_model["discriminatorState"]
		self.generator._load_model(generatorState)
		self.discriminator._load_model(discriminatorState)

		if "history_dict" in loaded_model:
			self.load_history(loaded_model["history_dict"])
			print("Succesfully loaded GAN model (with history, epoch %d)" % (self.currentEpoch))
		else:
			print("Succesfully loaded GAN model (no history)")
