import sys
import numpy as np

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

class MultiLinePrinter:
	def __init__(self):
		self.numLines = 0
		self.linePrinters = []

	def print(self, messages):
		if len(messages) > len(self.linePrinters):
			diff = len(messages) - len(self.linePrinters)
			for _ in range(diff):
				self.linePrinters.append(LinePrinter())

		# Print all N-1 lines with '\n' and last one strip '\n' if it exists
		linePrinters = self.linePrinters[0 : len(messages)]
		for i in range(len(messages) - 1):
			message = messages[i]
			if message[-1] != "\n":
				message += '\n'
			linePrinters[i].print(message)
		message = messages[-1]
		if message[-1] == "\n":
			message = message[0 : -1]
		linePrinters[-1].print(message)
		
		# Go up N-1 lines to overwrite at next message
		for i in range(len(messages) - 1):
			print("\033[1A", end="")
