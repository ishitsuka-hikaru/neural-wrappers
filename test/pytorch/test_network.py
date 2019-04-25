from neural_wrappers.pytorch import NeuralNetworkPyTorch, maybeCuda, maybeCpu
from neural_wrappers.callbacks import Callback, SaveModels, SaveHistory
import numpy as np
import torch as tr

import torch.nn as nn
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Model(NeuralNetworkPyTorch):
	def __init__(self, inputSize, hiddenSize, outputSize):
		super().__init__()
		self.fc1 = nn.Linear(inputSize, hiddenSize)
		self.fc2 = nn.Linear(hiddenSize, hiddenSize)
		self.fc3 = nn.Linear(hiddenSize, outputSize)

	def forward(self, x):
		y1 = self.fc1(x)
		y2 = self.fc2(y1)
		y3 = self.fc3(y2)
		return y3

class SchedulerCallback(Callback):
	def __init__(self, optimizer, **kwargs):
		super().__init__(**kwargs)
		self.scheduler = ReduceLROnPlateau(optimizer, "min", factor=0.1, patience=10, eps=1e-4)

	def onEpochEnd(self, **kwargs):
		if not kwargs["validationMetrics"]:
			loss = kwargs["trainMetrics"]["Loss"]
		else:
			loss = kwargs["validationMetrics"]["Loss"]
		self.scheduler.step(loss)

	def onCallbackLoad(self, additional, **kwargs):
		self.scheduler.optimizer = kwargs["model"].optimizer

class MyTestCallback(Callback):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)

class TestNetwork:
	def test_save_weights_1(self):
		N, I, H, O = 500, 100, 50, 30

		inputs = maybeCuda(tr.randn(N, I))
		targets = maybeCuda(tr.randn(N, O))
		model = maybeCuda(Model(I, H, O))
		for i in range(5):
			outputs = model.forward(inputs)
			error = tr.sum((outputs - targets)**2)

			error.backward()
			for param in model.parameters():
				param.data -= 0.001 * param.grad.data
				param.grad *= 0

		model.saveWeights("test_weights.pkl")
		model_new = maybeCuda(Model(I, H, O))
		model_new.loadWeights("test_weights.pkl")

		outputs = model.forward(inputs)
		error = maybeCpu(tr.sum((outputs - targets)**2)).data.numpy()
		outputs_new = model_new.forward(inputs)
		error_new = maybeCpu(tr.sum((outputs_new - targets)**2)).data.numpy()
		assert np.abs(error - error_new) < 1e-5

	def test_save_model_1(self):
		N, I, H, O = 500, 100, 50, 30
		inputs = np.float32(np.random.randn(N, I))	
		targets = np.float32(np.random.randn(N, O))

		model = maybeCuda(Model(I, H, O))
		model.setOptimizer(Adam, lr=0.001)
		model.setCriterion(lambda y, t : tr.sum((y - t)**2))

		model.train_model(data=inputs, labels=targets, batchSize=10, numEpochs=5, printMessage=False)
		model.saveModel("test_model.pkl")
		model.train_model(data=inputs, labels=targets, batchSize=10, numEpochs=5, printMessage=False)
		model_new = maybeCuda(Model(I, H, O))
		model_new.loadModel("test_model.pkl")
		model_new.setCriterion(lambda y, t : tr.sum((y - t)**2))
		model_new.train_model(data=inputs, labels=targets, batchSize=10, numEpochs=5, printMessage=False)

		weights_model = list(model.parameters())
		weights_model_new = list(model_new.parameters())

		assert len(weights_model) == len(weights_model_new)
		for j in range(len(weights_model)):
			weight = weights_model[j]
			weight_new = weights_model_new[j]
			diff = maybeCpu(tr.sum(tr.abs(weight - weight_new))).data.numpy()
			assert diff < 1e-5, "%d: Diff: %2.5f.\n %s %s" % (j, diff, weight, weight_new)

	def test_save_model_2(self):
		N, I, H, O = 500, 100, 50, 30
		inputs = np.float32(np.random.randn(N, I))
		targets = np.float32(np.random.randn(N, O))

		model = maybeCuda(Model(I, H, O))
		model.setOptimizer(SGD, lr=0.005)
		model.setCriterion(lambda y, t : tr.sum((y - t)**2))

		callbacks = [SchedulerCallback(model.optimizer, name="SchedulerCallback")]
		model.addCallbacks(callbacks)
		model.train_model(data=inputs, labels=targets, batchSize=10, numEpochs=10, printMessage=False)
		# print(model.callbacks[0].scheduler.num_bad_epochs)
		model.saveModel("test_model.pkl")
		model.train_model(data=inputs, labels=targets, batchSize=10, numEpochs=20, printMessage=False)
		# print(model.callbacks[0].scheduler.num_bad_epochs)
		assert model.callbacks["SchedulerCallback"].scheduler.optimizer == model.optimizer

		model_new = maybeCuda(Model(I, H, O))
		model_new.setCriterion(lambda y, t : tr.sum((y - t)**2))
		model_new.loadModel("test_model.pkl")
		# print(model_new.callbacks[0].scheduler.num_bad_epochs)
		assert model_new.callbacks["SchedulerCallback"].scheduler.optimizer == model_new.optimizer
		model_new.train_model(data=inputs, labels=targets, batchSize=10, numEpochs=20, printMessage=False)
		# print(model_new.callbacks[0].scheduler.num_bad_epochs)
		assert model.callbacks["SchedulerCallback"].scheduler.num_bad_epochs \
			== model_new.callbacks["SchedulerCallback"].scheduler.num_bad_epochs

		weights_model = list(model.parameters())
		weights_model_new = list(model_new.parameters())
		assert len(weights_model) == len(weights_model_new)
		for j in range(len(weights_model)):
			weight = weights_model[j]
			weight_new = weights_model_new[j]
			diff = maybeCpu(tr.sum(tr.abs(weight - weight_new))).data.numpy()
			assert diff < 1e-5, "%d: Diff: %2.5f.\n %s %s" % (j, diff, weight, weight_new)

	def test_save_model_3(self):
		N, I, H, O = 500, 100, 50, 30
		inputs = np.float32(np.random.randn(N, I))
		targets = np.float32(np.random.randn(N, O))

		model = maybeCuda(Model(I, H, O))
		model.setOptimizer(Adam, lr=0.01)
		model.setCriterion(lambda y, t: y.mean())

		callbacks = [SaveModels("best", "Loss", "min"), SaveModels("last"), SaveHistory("history.txt")]
		model.addCallbacks(callbacks)
		model.addMetrics({"Test" : lambda x, y, **k : 0.5})
		beforeKeys = list(model.callbacks.keys())
		model.train_model(data=inputs, labels=targets, batchSize=10, numEpochs=10, printMessage=False)
		model.saveModel("test_model.pkl")
		model.loadModel("test_model.pkl")
		afterKeys = list(model.callbacks.keys())

		assert len(beforeKeys) == len(afterKeys)
		for i in range(len(beforeKeys)):
			assert beforeKeys[i] == afterKeys[i]

	# Adding metrics normally should be fine.
	def test_add_merics_1(self):
		I, H, O = 100, 50, 30
		model = maybeCuda(Model(I, H, O))
		try:
			model.addMetrics({"Test" : lambda x, y, **k : 0.5})
		except Exception:
			assert False

		try:
			model.addMetrics({"Test2" : lambda x, y, **k : 0.5})
		except Exception as e:
			assert False

	# Adding two metrics with same name should clash it
	def test_add_merics_2(self):
		I, H, O = 100, 50, 30
		model = maybeCuda(Model(I, H, O))
		try:
			model.addMetrics({"Test" : lambda x, y, **k : 0.5})
		except Exception:
			assert False

		try:
			model.addMetrics({"Test" : lambda x, y, **k : 0.5})
		except Exception as e:
			return True
		assert False

	# Adding one metric and one callback with same name should clash it
	def test_add_merics_3(self):
		I, H, O = 100, 50, 30
		model = maybeCuda(Model(I, H, O))
		try:
			model.addMetrics({"Test" : lambda x, y, **k : 0.5})
		except Exception:
			assert False

		try:
			model.addCallbacks([MyTestCallback(name="Test")])
		except Exception:
			return True
		assert False

	def test_add_callbacks_1(self):
		I, H, O = 100, 50, 30
		model = maybeCuda(Model(I, H, O))
		try:
			callbacks = [SaveModels("best", "Loss", "min", name="SaveModelsBest"), \
				SaveModels("last", name="SaveModelsLast"), SaveHistory("history.txt")]
			model.addCallbacks([MyTestCallback(name="TestCallback")])
			model.addCallbacks(callbacks)
			model.addMetrics({"Test" : lambda x, y, **k : 0.5})
		except Exception:
			assert False

	def test_setCallbacksDependencies_1(self):
		I, H, O = 100, 50, 30
		model = maybeCuda(Model(I, H, O))
		try:
			callbacks = [SaveModels("best", "Loss", "min", name="SaveModelsBest"), \
				SaveModels("last", name="SaveModelsLast"), SaveHistory("history.txt")]
			model.addCallbacks([MyTestCallback(name="TestCallback")])
			model.addCallbacks(callbacks)
			model.addMetrics({"Test" : lambda x, y, **k : 0.5})
			model.setCallbacksDependencies({"TestCallback" : ["Test"], "SaveModelsLast" : ["SaveModelsBest"]})
		except Exception:
			assert False

	def test_setCallbacksDependencies_2(self):
		I, H, O = 100, 50, 30
		model = maybeCuda(Model(I, H, O))
		try:
			callbacks = [SaveModels("best", "Loss", "min", name="SaveModelsBest"), \
				SaveModels("last", name="SaveModelsLast"), SaveHistory("history.txt")]
			model.addCallbacks([MyTestCallback(name="TestCallback")])
			model.addCallbacks(callbacks)
			model.addMetrics({"Test" : lambda x, y, **k : 0.5})
		except Exception:
			assert False

		try:
			model.setCallbacksDependencies({"NonExistant" : ["Test"], "SaveModelsLast" : ["SaveModelsBest"]})
		except Exception:
			pass

		try:
			model.setCallbacksDependencies({"Test" : ["NonExistant"], "SaveModelsLast" : ["SaveModelsBest"]})
		except Exception:
			pass

	def test_setCallbacksDependencies_3(self):
		I, H, O = 100, 50, 30
		model = maybeCuda(Model(I, H, O))
		try:
			callbacks = [SaveModels("best", "Loss", "min", name="SaveModelsBest"), \
				SaveModels("last", name="SaveModelsLast"), SaveHistory("history.txt")]
			model.addCallbacks([MyTestCallback(name="TestCallback")])
			model.addCallbacks(callbacks)
			model.addMetrics({"Test" : lambda x, y, **k : 0.5})
		except Exception:
			assert False
		
		try:
			model.setCallbacksDependencies({"TestCallback" : ["Test"], "SaveModelsLast" : ["SaveModelsLast"]})
		except Exception:
			pass

		try:
			model.setCallbacksDependencies({"TestCallback" : ["Test"], "Test" : ["TestCallback"]})
		except Exception:
			pass

		try:
			model.setCallbacksDependencies({"TestCallback" : ["Test"], "Test" : ["SaveModelsBest"], \
				"SaveModelsBest" : ["TestCallback"], "SaveModelsLast" : ["SaveModelsBest"]})
		except Exception:
			pass

		try:
			model.setCallbacksDependencies({"TestCallback" : ["Test"], "Test" : ["SaveModelsLast", "Loss"], \
				"SaveModelsLast" : ["SaveModelsBest"]})
		except Exception:
			assert False

	def test_setCallbacksDependencies_4(self):
		I, H, O = 100, 50, 30
		model = maybeCuda(Model(I, H, O))
		model_new = maybeCuda(Model(I, H, O))
		try:
			callbacks = [SaveModels("best", "Loss", "min", name="SaveModelsBest"), \
				SaveModels("last", name="SaveModelsLast"), SaveHistory("history.txt")]
			model.addCallbacks([MyTestCallback(name="TestCallback")])
			model.addCallbacks(callbacks)
			model.addMetrics({"Test" : lambda x, y, **k : 0.5})
			model.setCallbacksDependencies({"TestCallback" : ["Test"], "Test" : ["SaveModelsLast", "Loss"], \
				"SaveModelsLast" : ["SaveModelsBest"]})
			model_new.addMetrics({"Test" : lambda x, y, **k : 0.5})
		except Exception:
			assert False
		model.setOptimizer(SGD, lr=0.005)
		model.saveModel("test_model.pkl")
		model_new.loadModel("test_model.pkl")

		callbackKeys = list(model.callbacks.keys())
		callbackKeysNew = list(model_new.callbacks.keys())
		assert len(callbackKeys) == len(callbackKeysNew)
		assert len(model.topologicalSort) == len(model_new.topologicalSort)
		for i in range(len(callbackKeys)):
			assert callbackKeys[i] == callbackKeysNew[i]
			assert model.topologicalSort[i] == model_new.topologicalSort[i]
			assert model.topologicalKeys[i] == model_new.topologicalKeys[i]

if __name__ == "__main__":
	TestNetwork().test_save_weights_1()
	TestNetwork().test_save_model_1()
	TestNetwork().test_save_model_2()
	TestNetwork().test_save_model_3()
	TestNetwork().test_add_merics_1()
	TestNetwork().test_add_merics_2()
	TestNetwork().test_add_merics_3()
	TestNetwork().test_add_callbacks_1()
	TestNetwork().test_setCallbacksDependencies_1()
	TestNetwork().test_setCallbacksDependencies_2()
	TestNetwork().test_setCallbacksDependencies_3()
	TestNetwork().test_setCallbacksDependencies_4()