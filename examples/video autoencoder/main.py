import sys
import numpy as np
import torch.nn as nn
import torch as tr
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from wrappers.pytorch import RecurrentNeuralNetworkPyTorch, maybeCuda, maybeCpu
from models.resnet50_notop import ResNet50NoTop
from scipy.misc import toimage
from Mihlib import readVideo, saveVideo
from callbacks import SaveModels
from functools import partial

class RecurrentCNN(RecurrentNeuralNetworkPyTorch):
	def __init__(self, inputSize=(240, 320, 3), hiddenSize=300, outputSize=(30, 50, 3)):
		super().__init__()
		self.inputSize = inputSize
		self.hiddenSize = hiddenSize
		self.outputSize = outputSize
		# self.baseModelType = baseModelType

		# Feature extractor part
		baseOutputShape = {
			240 : 8,
			320 : 10,
			360 : 12,
			480 : 15
		}
		avgPoolShape = (baseOutputShape[inputSize[0]], baseOutputShape[inputSize[1]])
		self.lstmInputShape = 2048

		self.baseModel = ResNet50NoTop()
		self.avgpool = nn.AvgPool2d(avgPoolShape, stride=1)

		# LSTM part
		self.lstm1 = nn.LSTM(input_size=self.lstmInputShape, hidden_size=hiddenSize, num_layers=1)
		self.fc1 = nn.Linear(in_features=hiddenSize, out_features=int(np.prod(outputSize)))

	def forward(self, input, hidden):
		input = tr.transpose(tr.transpose(input, 1, 3), 2, 3) # Move depth first
		miniBatch = input.shape[0]
		y1 = self.baseModel(input)
		y2 = self.avgpool(y1)
		y2 = y2.view(miniBatch, 1, self.lstmInputShape).contiguous()
		# self.lstm1.flatten_parameters()
		y3, h = self.lstm1(y2, hidden[0])
		y4 = self.fc1(y3)
		# MBx4500 => MBx30x50x3 (for example)
		y4 = y4.view(miniBatch, *self.outputSize)
		return y4, [h]

def videoGenerator(video, label, numEpochs, numFrames, sequenceSize):
	numSequences = numFrames // sequenceSize + (numFrames % sequenceSize != 0)
	for e in range(numEpochs):
		for j in range(numSequences):
			seqStartIndex, seqEndIndex = j * sequenceSize, np.minimum((j + 1) * sequenceSize, numFrames)
			currentVideoSeq = np.float32(np.expand_dims(np.array(video[seqStartIndex : seqEndIndex]), axis=0) / 255)
			currentLabelSeq = np.float32(np.expand_dims(np.array(label[seqStartIndex : seqEndIndex]), axis=0) / 255)
			yield currentVideoSeq, currentLabelSeq

# At each step during one epoch, get each N : N + s frames, and store them
def storeFrames(finalArray, sequenceSize, **kwargs):
	results = kwargs["results"]
	iteration = kwargs["iteration"]
	startIndex, endIndex = sequenceSize * iteration, (sequenceSize * iteration) + results.shape[1]
	print("%d:%d out of %d" % (startIndex, endIndex, sequenceSize * kwargs["numIterations"]))

	frames = list(map(lambda x : np.array(toimage(x)), results[0]))
	finalArray[startIndex : endIndex] = frames

def main():
	assert sys.argv[1] in ("train", "test")

	video, label = readVideo(sys.argv[2]), readVideo(sys.argv[3])
	assert len(video) == len(label)
	numFrames = len(video)
	sequenceSize = 5

	if sys.argv[1] == "train":
		assert len(sys.argv) == 4
		model = maybeCuda(RecurrentCNN(inputSize=video.frame_shape, hiddenSize=300, outputSize=label.frame_shape))
		print(model.summary())

		generator = videoGenerator(video, label, numEpochs=10, numFrames=numFrames, sequenceSize=sequenceSize)
		numSequences = numFrames // sequenceSize + (numFrames % sequenceSize != 0)
		print("Video shape: %s. Label shape: %s. Num frames: %d" % (video.frame_shape, label.frame_shape, numFrames))

		model.setOptimizer(optim.Adam, lr=0.1)
		model.setCriterion(lambda x, y : tr.mean((x - y)**2))
		model.train_generator(generator, stepsPerEpoch=numSequences, numEpochs=10, callbacks=[SaveModels(type="all")])
	elif sys.argv[1] == "test":
		assert len(sys.argv) == 5
		# Here label is used just to extract the correct shape
		video, label = readVideo(sys.argv[2]), readVideo(sys.argv[3])
		model = maybeCuda(RecurrentCNN(inputSize=video.frame_shape, hiddenSize=300, outputSize=label.frame_shape))
		model.load_weights(sys.argv[4])
		model.setCriterion(lambda x, y : tr.mean((x - y)**2))

		numFrames = len(video) // 2
		result = np.zeros(shape=(numFrames, *label.frame_shape), dtype=np.uint8)
		storeFramesCallback = partial(storeFrames, finalArray=result, sequenceSize=sequenceSize)
		generator = videoGenerator(video, label, numEpochs=10, numFrames=numFrames, sequenceSize=sequenceSize)
		numSequences = numFrames // sequenceSize + (numFrames % sequenceSize != 0)

		model.test_generator(generator, numSequences, callbacks=[storeFramesCallback])
		saveVideo(npData=result, fileName="video_result.mp4", fps=30)

if __name__ == "__main__":
	main()