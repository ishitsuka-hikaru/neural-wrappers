import numpy as np
import sys
# from lycon import resize, Interpolation
import cv2
from scipy.ndimage import gaussian_filter

def resize_batch(data, dataShape, type="bilinear"):
        # No need to do anything if shapes are identical.
        if data.shape[1 : ] == dataShape:
                return np.copy(data)

        assert type in ("bilinear", "nearest", "cubic")
        if type == "bilinear":
                # interpolationType = Interpolation.LINEAR
                interpolationType = cv2.INTER_LINEAR
        elif type == "nearest":
                # interpolationType = Interpolation.NEAREST
                interpolationType = cv2.INTER_NEAREST
        else:
                # interpolationType = Interpolation.CUBIC
                interpolationType = cv2.INTER_CUBIC

        numData = len(data)
        newData = np.zeros((numData, *dataShape), dtype=data.dtype)

        for i in range(len(data)):
                # result = resize(data[i], height=dataShape[0], width=dataShape[1], interpolation=interpolationType)
                result = cv2.resize(data[i], (dataShape[1], dataShape[0]), interpolation=interpotationType)
                newData[i] = result.reshape(newData[i].shape)
        return newData

def resize_black_bars(data, desiredShape, type="bilinear"):
        # No need to do anything if shapes are identical.
        if data.shape == desiredShape:
                return np.copy(data)

        assert type in ("bilinear", "nearest", "cubic")
        if type == "bilinear":
                interpolationType = Interpolation.LINEAR
        elif type == "nearest":
                interpolationType = Interpolation.NEAREST
        else:
                interpolationType = Interpolation.CUBIC

        newData = np.zeros(desiredShape, dtype=data.dtype)
        # newImage = np.zeros((240, 320, 3), np.uint8)
        h, w = data.shape[0 : 2]
        desiredH, desiredW = desiredShape[0 : 2]

        # Find the rapports between the h/desiredH and w/desiredW
        rH, rW = h / desiredH, w / desiredW
        # print(rH, rW)

        # Find which one is the highest, that one will be used
        minRapp, maxRapp = min(rH, rW), max(rH, rW)
        # print(minRapp, maxRapp)

        # Compute the new dimensions, based on th highest rapport
        newRh, newRw = int(h // maxRapp), int(w // maxRapp)
        # Also, find the half, so we can inser the other dimension from the half
        halfH, halfW = int((desiredH - newRh) // 2), int((desiredW - newRw) // 2)

        resizedData = resize(data, height=newRh, width=newRw, interpolation=interpolationType)
        newData[halfH : halfH + newRh, halfW : halfW + newRw] = resizedData
        return newData

# Resizes a batch of HxW images, to a desired dHxdW, but keeps the same aspect ration, and adds black bars on the
#  dimension that does not fit (instead of streching as with regular resize).
def resize_batch_black_bars(data, desiredShape, type="bilinear"):
        # No need to do anything if shapes are identical.
        if data.shape[1 : ] == desiredShape:
                return np.copy(data)

        newData = np.zeros((numData, *desiredShape), dtype=data.dtype)
        for i in range(len(data)):
                newData[i] = resize_black_bars(data[i], desiredShape, type)

        return newData

def standardizeData(data, mean, std):
        data -= mean
        data /= std
        return data

def minMaxNormalizeData(data, min, max):
        data -= min
        data /= (max - min)
        return data

def toCategorical(data, numClasses):
        numData = len(data)
        newData = np.zeros((numData, numClasses), dtype=np.uint8)
        newData[np.arange(numData), data] = 1
        return newData

# Labels can be None, in that case only data is available (testing cases without labels)
def makeGenerator(data, labels, batchSize):
        while True:
                numData = data.shape[0]
                numIterations = numData // batchSize + (numData % batchSize != 0)
                for i in range(numIterations):
                        startIndex = i * batchSize
                        endIndex = np.minimum((i + 1) * batchSize, numData)
                        if not labels is None:
                                yield data[startIndex : endIndex], labels[startIndex : endIndex]
                        else:
                                yield data[startIndex : endIndex]

def NoneAssert(conndition, noneCheck, message=""):
        if noneCheck:
                assert conndition, message

class LinePrinter:
        def __init__(self):
                self.maxLength = 0

        def print(self, message):
                if message[-1] == "\n":
                        message = message[0 : -1]
                        additional = "\n"
                else:
                        additional = "\r"

                self.maxLength = np.maximum(len(message), self.maxLength)
                message += (self.maxLength - len(message)) * " " + additional
                sys.stdout.write(message)
                sys.stdout.flush()

# @brief Returns true if whatType is subclass of baseType. The parameters can be instantiated objects or types. In the
#  first case, the parameters are converted to their type and then the check is done.
def isBaseOf(whatType, baseType):
        if type(whatType) != type:
                whatType = type(whatType)
        if type(baseType) != type:
                baseType = type(baseType)
        return baseType in type(object).mro(whatType)

# Stubs for identity functions, first is used for 1 parameter f(x) = x, second is used for more than one parameter,
#  such as f(x, y, z) = (x, y, z)
def identity(x, **kwargs):
        return x

def identityVar(*args):
        return args

# Stub for making a list, used by various code parts, where the user may provide a single element for a use-case where
#  he'd have to use a 1-element list. This handles that case, so the overall API uses lists, but user provides
#  just an element. If None, just return None.
def makeList(x):
        return None if type(x) == type(None) else x if type(x) == list else [x]

# ["test"] and ["blah", "blah2"] => False
# ["blah2"] and ["blah", "blah2"] => True
def isSubsetOf(subset, set):
        for item in subset:
                if not item in set:
                        return False
        return True
