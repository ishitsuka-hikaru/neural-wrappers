import numpy as np

def minMaxImage(image):
	Min, Max = image.min(), image.max()
	return (image - Min) / (Max - Min)