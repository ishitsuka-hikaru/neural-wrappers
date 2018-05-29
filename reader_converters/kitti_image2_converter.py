import h5py
import sys, os
import cv2
# from PIL import Image
import numpy as np
from lycon import resize, Interpolation

'''
Final structure:
- train
    camera_2
      rgb
      labels
        label1
        ...
        labelN
      calibration
- test
    camera_2
      rgb
      calibration
'''


def flushPrint(message):
	sys.stdout.write(message + "\n")
	sys.stdout.flush()

def getPaths(basePath, labelData):
	paths = {}
	rgbFiles = sorted(os.listdir(basePath + os.sep + "image_2"))
	if labelData:
		paths["label"] = list(map(lambda x : basePath + os.sep + "label_2" + os.sep + x[0 : -4] + ".txt", rgbFiles))
	paths["calib"] = list(map(lambda x : basePath + os.sep + "calib" + os.sep + x[0 : -4] + ".txt", rgbFiles))
	paths["rgb"] = list(map(lambda x : basePath + os.sep + "image_2" + os.sep + x, rgbFiles))
	return paths

def doPng(pngPath):
	# image = np.uint8(Image.open(pngPath))
	bgr_image = cv2.imread(pngPath)
	try:
		b, g, r = cv2.split(bgr_image)
		image = cv2.merge([r, g, b]).astype(np.float32)
		return resize(image, height=375, width=1224, interpolation=Interpolation.LANCZOS)
	except Exception:
		sys.stdout.write("Error at %s\n" % (pngPath))
		sys.flush()

def doLabel(labelPath):
	classes = ["Pedestrian", "Truck", "Car", "Cyclist", "Misc", "Van", "Tram", "Person_sitting"]
	f = open(labelPath)
	lines = f.readlines()
	numLines = len(lines)
	labels = []

	for i in range(numLines):
		line = lines[i]
		object_label = line.split(" ")
		if not object_label[0] in classes:
			continue
		object_label[0] = classes.index(object_label[0])
		object_label[1 : ] = list(map(lambda x : float(x), object_label[1 : ]))
		labels.append(object_label)
	labels = np.array(labels, dtype=np.float32)
	f.close()
	# Labels are flattened at end, because they must be stored as 1D variable length array
	return labels.flatten()

def doCalibration(calibPath):
	f = open(calibPath)
	lines = f.readlines()
	for line in lines:
		if not line.startswith("P2"):
			continue
		items = line.strip().split(" ")[1 : ]
		calibration = np.asarray([float(number) for number in items]).reshape((3, 4))
	f.close()
	return calibration

def doH5pyStuff(file, groupName, labelData, dataShape):
	if groupName in file:
		group = file[groupName]
	else:
		group = file.create_group(groupName)

	numData = dataShape[0]
	if not "camera_2" in group:
		group = group.create_group("camera_2")
	if not "rgb" in group:
		group.create_dataset("rgb", shape=dataShape, dtype=np.uint8)
	if labelData:
		if not "labels" in group:
			group.create_dataset("labels", shape=(numData, ), dtype=h5py.special_dtype(vlen=np.float32))
	if not "calibration" in group:
		group.create_dataset("calibration", shape=(numData, 3, 4))
	return group

def doDataset(file, basePath, groupName, labelData, startIndex):
	paths = getPaths(basePath, labelData)
	numData = len(paths["rgb"])
	dataShape = (numData, 375, 1224, 3)
	group = doH5pyStuff(file, groupName, labelData, dataShape)

	endIndex = min(startIndex + 100, numData)
	if startIndex > endIndex:
		return

	for i in range(startIndex, endIndex):
		if i % 100 == 0:
			print("%d/%d done" % (i, numData))
		image = doPng(paths["rgb"][i])
		calib = doCalibration(paths["calib"][i])
		group["rgb"][i] = image
		group["calibration"][i] = calib
		if labelData:
			label = doLabel(paths["label"][i])
			group["labels"][i] = label

def main():
	assert len(sys.argv) == 3
	file = h5py.File("kitti_obj.h5", "a")

	baseDir = os.path.abspath(sys.argv[1])
	doDataset(file, baseDir + os.sep + "training", "train", labelData=True, startIndex=int(sys.argv[2]))
	doDataset(file, baseDir + os.sep + "testing", "test", labelData=False, startIndex=int(sys.argv[2]))

if __name__ == "__main__":
	main()