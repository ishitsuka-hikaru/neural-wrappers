import torch.nn as nn
import torch.nn.functional as F
from neural_wrappers.pytorch import FeedForwardNetwork

class Model(FeedForwardNetwork):
	def __init__(self, cellType, inputSize, hiddenSize, outputSize):
		super(Model, self).__init__()
		assert cellType in ("RNN", "GRU", "LSTM")
		self.hiddenSize = hiddenSize
		self.inputSize = inputSize
		self.outputSize = outputSize
		self.cellType = cellType

		if self.cellType == "RNN":
			self.rnn = nn.RNN(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=1)
		elif self.cellType == "GRU":
			self.rnn = nn.GRU(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=1)
		elif self.cellType == "LSTM":
			self.rnn = nn.GRU(input_size=self.inputSize, hidden_size=self.hiddenSize, num_layers=1)
		self.fc1 = nn.Linear(self.hiddenSize, self.outputSize)

	def forward(self, input):
		if isinstance(input, (tuple, list)):
			input, hidden = input
		else:
			hidden = None

		output, hidden = self.rnn(input, hidden)
		output = self.fc1(output)
		output = F.softmax(output, dim=-1)
		return output, hidden
