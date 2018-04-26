from datetime import datetime

from torch.autograd import Variable
import torch.nn as nn
import torch as tr

from .utils import maybeCuda, maybeCpu
from .network import NeuralNetworkPyTorch
from neural_wrappers.utilities import LinePrinter

class GenerativeAdversialNetwork(NeuralNetworkPyTorch):
	def __init__(self, generator, discriminator):
		super().__init__()
		self.generator = generator.cuda()
		self.discriminator = discriminator.cuda()
		self.criterion = tr.nn.BCELoss().cuda()
		self.linePrinter = LinePrinter()

	def run_one_epoch(self, generator, stepsPerEpoch):
		npLossD, npLossG = 0, 0
		startTime = datetime.now()
		for i in range(stepsPerEpoch):
			imgs, _ = next(generator)
			imgs = tr.from_numpy(imgs)

			# Adversarial ground truths
			fake = Variable(tr.zeros(imgs.shape[0]).cuda(), requires_grad=False)
			valid = Variable(tr.ones(imgs.shape[0]).cuda(), requires_grad=False)

			# Configure input
			real_imgs = Variable(maybeCuda(imgs))

			#  Train Generator
			# -----------------

			self.generator.optimizer.zero_grad()

			# Sample noise as generator input
			z = Variable(tr.randn(imgs.shape[0], 100).cuda())

			# Generate a batch of images
			gen_imgs = self.generator(z)

			# Loss measures generator's ability to fool the discriminator
			g_loss = self.criterion(self.discriminator(gen_imgs), valid)
			npLossG += maybeCpu(g_loss.data).numpy()

			g_loss.backward()
			self.generator.optimizer.step()

			# ---------------------
			#  Train Discriminator
			# ---------------------

			self.discriminator.optimizer.zero_grad()

			# Measure discriminator's ability to classify real from generated samples
			real_loss = self.criterion(self.discriminator(real_imgs), valid)
			fake_loss = self.criterion(self.discriminator(gen_imgs.detach()), fake)
			d_loss = (real_loss + fake_loss) / 2
			npLossD += maybeCpu(d_loss.data).numpy()

			d_loss.backward()
			self.discriminator.optimizer.step()

			iterFinishTime = (datetime.now() - startTime)
			ETA = abs(iterFinishTime / (i + 1) * stepsPerEpoch - iterFinishTime)
			self.linePrinter.print("Epoch: %d/%d. Iteration: %d/%d Loss D: %2.2f. Loss G: %2.2f. ETA: %s" % \
				(self.currentEpoch, self.numEpochs, i, stepsPerEpoch, npLossD / (i + 1), npLossG / (i + 1), ETA))

		return npLossD / stepsPerEpoch, npLossG / stepsPerEpoch

	def train_generator(self, generator, stepsPerEpoch, numEpochs, callbacks=[]):
		self.checkCallbacks(callbacks)
		self.numEpochs = numEpochs

		self.linePrinter.print("Training for %d epochs...\n" % (numEpochs))
		while self.currentEpoch < numEpochs + 1:
			self.trainHistory.append({})

			now = datetime.now()
			npLossD, npLossG = self.run_one_epoch(generator, stepsPerEpoch)
			duration = datetime.now() - now

			self.linePrinter.print("Epoch: %d/%d. Loss D: %2.2f. Loss G: %2.2f. Took %s\n" % \
				(self.currentEpoch, numEpochs, npLossD, npLossG, duration))

			self.trainHistory[self.currentEpoch - 1] = {
				"generatorLoss" : npLossG,
				"discriminatorLoss" : npLossD
			}

			# Do the callbacks for the end of epoch.
			callbackArgs = {
				"model" : self,
				"epoch" : self.currentEpoch,
				"numEpochs" : numEpochs,
				"duration" : duration,
				"trainHistory" : self.trainHistory[self.currentEpoch - 1],
				"trainMetrics": None,
				"validationMetrics" : None
			}
			for callback in callbacks:
				callback.onEpochEnd(**callbackArgs)

			self.currentEpoch += 1

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