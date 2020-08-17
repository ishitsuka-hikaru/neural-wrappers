import sys
import numpy as np
from overrides import overrides
from abc import ABC, abstractmethod

class MessagePrinter(ABC):
	def __init__(self):
		self.printer = None

	def getPrinter(Type):
		return {
			None : NonePrinter,
			"none" : NonePrinter,
			"v1" : LinePrinter,
			"v2" : MultiLinePrinter
		}[Type]()

	@abstractmethod
	def print(self, message, **kwargs):
		pass

	def __call__(self, message, **kwargs):
		return self.print(message, **kwargs)

class NonePrinter(MessagePrinter):
	@overrides
	def print(self, message, **kwargs):
		pass

# Class that prints one line to the screen. Appends "\r" if no ending character is found and also appends white chars
#  so printing between two iterations don't mess up the screen.
class LinePrinter(MessagePrinter):
	def __init__(self):
		self.maxLength = 0

	@overrides
	def print(self, message, reset=True):
		if type(message) in (list, tuple):
			message = ". ".join(message)
		message = message.replace("\n", "").replace("  ", " ").replace("..", ".")

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
class MultiLinePrinter(MessagePrinter):
	def __init__(self):
		self.numLines = 0
		self.linePrinters = []

	@overrides
	def print(self, messages, reset=True):
		if type(messages) == str:
			messages = [messages]

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