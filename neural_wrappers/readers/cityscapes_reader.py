import h5py
import numpy as np
import os
import pims
from .dataset_reader import DatasetReader
from neural_wrappers.transforms import Transformer

class CityScapesReader(DatasetReader):
	# @param skipFrames Each video is consistent of multiple 30-frames scenes (1 second at 30fps). For videos, we want
	#  to process every frame (so skipFrames is 1 by default). For some other type of processing (like training a
	#  non-recurrent neural network), we might want to skip some frames (like get only every nth frame), so this value
	#  is higher. It cannot exceed 30, because there are only 30 frames in every scene and each scene is an independent
	#  unit of work.
	def __init__(self, datasetPath, imageShape, labelShape, skipFrames=1, dataSplit=(80, 0, 20)):
		assert skipFrames > 0 and skipFrames <= 30
		self.datasetPath = datasetPath
		self.imageShape = imageShape
		self.labelShape = labelShape
		self.dataSplit = dataSplit
		self.dataAugmenter = Transformer(["none"], dataShape=imageShape, labelShape=labelShape)
		self.skipFrames = skipFrames
		self.setup()

	def setup(self):
		videosPath = self.datasetPath + os.sep + "video"
		depthPath = self.datasetPath + os.sep + "depth"
		flowFarnebackPath = self.datasetPath + os.sep + "flow_farneback"
		flowDeepFlow2Path = self.datasetPath + os.sep + "flow_deepflow2"

		allVideos = os.listdir(videosPath)
		allVideos = sorted(list(filter(lambda x : x.endswith("mp4"), allVideos)))
		numVideos = len(allVideos)
		assert numVideos > 0

		self.indexes, self.numData = self.computeIndexesSplit(numVideos)

		# Using the indexes, find all the paths to all the videos of all types for all labels. These will be used
		#  during iteration to actually read the videos and access the frame for training/testing.
		self.paths = {
			"train" : {
				"video" : [],
				"depth" : [],
				"flow_farnaback" : [],
				"flow_deepflow2" : []
			},
			"test" : {
				"video" : [],
				"depth" : [],
				"flow_farnaback" : [],
				"flow_deepflow2" : []
			},
			"validation" : {
				"video" : [],
				"depth" : [],
				"flow_farnaback" : [],
				"flow_deepflow2" : []
			}
		}

		self.totalDurations = {
			"train" : 0,
			"test" : 0,
			"validation" : 0
		}

		self.durations = {
			"train" : [],
			"test" : [],
			"validation" : []
		}

		for Type in ["train", "test", "validation"]:
			typeVideos = allVideos[self.indexes[Type][0] : self.indexes[Type][1]]
			for i in range(len(typeVideos)):
				videoName = typeVideos[i]
				depthName = depthPath + os.sep + "depth_" + videoName[6 : ]
				flowFarnebackName = flowFarnebackPath + os.sep + "flow_farneback_" + videoName[6 : -4] + ".avi"
				flowDeepFlow2Name = flowDeepFlow2Path + os.sep + "flow_deepflow2_" + videoName[6 : -4] + ".avi"
				videoName = videosPath + os.sep + videoName

				self.paths[Type]["video"].append(videoName)
				self.paths[Type]["depth"].append(depthName)
				self.paths[Type]["flow_farnaback"].append(flowFarnebackName)
				self.paths[Type]["flow_deepflow2"].append(videoName)

				# Also save the duration, so we can compute the number of iterations (Takes about 10s for all dataset)
				video = pims.Video(videoName)
				# Some videos have 601 frames, instead of 600, so remove that last one (also during iteration)
				videoLen = len(video) - (len(video) % 30)
				self.totalDurations[Type] += videoLen
				self.durations[Type].append(videoLen)

		print(("[CityScapes Reader] Setup complete. Num videos: %d. Train: %d, Test: %d, Validation: %d. " + \
			"Frame shape: %s. Labels shape: %s. Frames: Train: %d, Test: %d, Validation: %d.") % (numVideos, \
			self.numData["train"], self.numData["test"], self.numData["validation"], self.imageShape, \
			self.labelShape, self.totalDurations["train"], self.totalDurations["test"], \
			self.totalDurations["validation"]))

	def getNumIterations(self, type, miniBatchSize, accountTransforms=False):
		assert miniBatchSize > 0 and miniBatchSize <= 20
		typeDurations = self.durations[type]
		numIterationsPerBatch = int(np.ceil(30 / self.skipFrames))
		numIterations = 0

		# For each clip (600, 600, 450, ... 30 etc.), find how many actual iterations we have. Each 30-frames scene
		#  is considered an individual item in the batch. The last batch however it's possible to have less scenes
		#  due to missmatch between miniBatchSize (i.e. 450-frames with mb=7 has only 15 scenes (15*30=450), so we have
		#  two sets of full batches and the last batch contains just 1 scene (15-7*2). Each frame is an iteration,
		#  regardless of the size of the miniBatch (fullBatch of 7 or just the last batch of 1 scene).
		for i in range(len(typeDurations)):
			clip = typeDurations[i]
			numScenes = clip // 30
			N = numScenes // miniBatchSize + (numScenes % miniBatchSize != 0)
			numIterations += N * numIterationsPerBatch

			# Last batch could be complete (divides exactly) or partial.
			# lastBatchContribution = numScenes % miniBatchSize if numScenes % miniBatchSize != 0 else miniBatchSize
			# firstBatchesContribution = (N - 1) * miniBatchSize * numIterationsPerBatch
			# lastBatchContribution *= numIterationsPerBatch
			# numIterations += firstBatchesContribution + lastBatchContribution

		if accountTransforms:
			numIterations *= len(self.transforms)
		return numIterations

	def iterate_once(self, type, miniBatchSize):
		assert type in ("train", "test", "validation")
		augmenter = self.dataAugmenter if type == "train" else self.validationAugmenter

		numVideos = self.numData[type]
		paths = self.paths[type]
		# This number is constant for every miniBatch and varies from 1 to 30 (30 for full videos).
		numIterationsPerBatch = int(np.ceil(30 / self.skipFrames))

		for video_i in range(numVideos):
			video = pims.Video(paths["video"][video_i])
			depthVideo = pims.Video(paths["depth"][video_i])
			# flowFarnabackVideo = pims.Video(paths["flow_farnaback"][i])
			# flowDeepflow2Video = pims.Video(paths["flow_deepflow2"][i])
			videoLen = len(video) - (len(video) % 30)

			clip = self.durations[type][video_i]
			numScenes = clip // 30
			N = numScenes // miniBatchSize + (numScenes % miniBatchSize != 0)
			lastBatchContribution = numScenes % miniBatchSize if numScenes % miniBatchSize != 0 else miniBatchSize

			# First N - 1 batches are complete
			for batch_i in range(N):
				startIndex = batch_i * miniBatchSize * 30
				endIndex = min((batch_i + 1) * miniBatchSize * 30, videoLen)
				# batchIndexes = [0, 30, 60, 90]. After one step: [1, 31, 61, 91] (for batchSize=4, skipFrames=1)
				batchIndexes = np.arange(startIndex, endIndex, 30)
				assert (len(batchIndexes) == miniBatchSize and batch_i < N - 1) or \
					(len(batchIndexes) == lastBatchContribution and batch_i == N - 1)
				for batch_step_i in range(numIterationsPerBatch):
					images = video[batchIndexes]
					depths = depthVideo[batchIndexes]
					# if batch_step_i == 29 => repeat previous one for flow!
					yield images, depths, None, None
					batchIndexes += self.skipFrames