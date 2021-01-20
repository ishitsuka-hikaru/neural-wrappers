import sys
import numpy as np
from typing import Union
from ...utilities import NWNumber, NWSequence, NWDict, RunningMean

def getFormattedStr(item : Union[np.ndarray, NWNumber, NWSequence, NWDict], precision : int) -> str: \
	# type: ignore
	formatStr = "%%2.%df" % (precision)
	if type(item) in NWNumber.__args__: # type: ignore
		return formatStr % (item) # type: ignore
	elif type(item) in NWSequence.__args__: # type: ignore
		return [formatStr % (x) for x in item] # type: ignore
	elif type(item) in NWDict.__args__: # type: ignore
		return {k : formatStr % (item[k]) for k in item} # type: ignore
	elif isinstance(item, RunningMean):
		return getFormattedStr(item.get(), precision)
	assert False, "Unknown type: %s" % (type(item))

class MessagePrinter:
	def __init__(self, type):
		assert type in (None, "v1", "v2", "tqdm")

		if type == None:
			self.printer = NonePrinter()
			self.function = lambda x : x
		elif type == "v1":
			self.printer = LinePrinter()
			self.function = MessagePrinter.printV1
		elif type == "v2":
			self.printer = MultiLinePrinter()
			self.function = MessagePrinter.printV2
		elif type == "tqdm":
			self.printer = NonePrinter()
			self.function = lambda x : x

	# For V1 printing, if we receive a list of messages, concatenate them by ". " and send them to line printer.
	def printV1(message):
		if type(message) in (list, tuple):
			message = ". ".join(message)
		message = message.replace("\n", "").replace("  ", " ").replace("..", ".")
		return message

	def printV2(message):
		if type(message) == str:
			message = [message]
		return message

	def print(self, message, **kwargs):
		self.printer.print(self.function(message), **kwargs)

	def __call__(self, message, **kwargs):
		return self.print(message, **kwargs)

class NonePrinter:
	def print(self, message, **kwargs):
		pass

# Class that prints one line to the screen. Appends "\r" if no ending character is found and also appends white chars
#  so printing between two iterations don't mess up the screen.
class LinePrinter:
	def __init__(self):
		self.maxLength = 0

	def print(self, message, reset=True):
		if len(message) == 0:
			message = ""
			additional = "\n"
		elif message[-1] == "\n":
			message = message[0 : -1]
			additional = "\n"
		else:
			additional = "\r"

		self.maxLength = np.maximum(len(message), self.maxLength)
		message += (self.maxLength - len(message)) * " " + additional
		sys.stdout.write(message)
		sys.stdout.flush()

		if not reset:
			print("")

# Class that prints multiple lines to the screen.
class MultiLinePrinter:
	def __init__(self):
		self.numLines = 0
		self.linePrinters = []

	def print(self, messages, reset=True):
		if len(messages) > len(self.linePrinters):
			diff = len(messages) - len(self.linePrinters)
			for _ in range(diff):
				self.linePrinters.append(LinePrinter())

		# Print all N-1 lines with '\n' and last one strip '\n' if it exists
		linePrinters = self.linePrinters[0 : len(messages)]
		for i in range(len(messages) - 1):
			message = messages[i]
			if message[-1] != "\n":
				message += "\n"
			linePrinters[i].print(message)
		message = messages[-1]
		if message[-1] == "\n":
			message = message[0 : -1]
		linePrinters[-1].print(message)
		
		if reset:
			# Go up N-1 lines to overwrite at next message
			for i in range(len(messages) - 1):
				print("\033[1A", end="")
		else:
			print("")