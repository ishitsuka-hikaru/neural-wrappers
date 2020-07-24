from __future__ import annotations
from typing import Tuple, Union, Any
from ..utilities import isBaseOf

# Wrapper on top of callback names. This is to reduce overhead of checking if a metric name is tuple or string, since it
#  can be both, depending on context. Graph metrics are stored as a tuple of strings.
class CallbackName:
    def __init__(self, name:Union[str, Tuple[Any]]):
        if isBaseOf(name, str):
            name = (name, )
        self.name = name

    def __str__(self) -> str:
        if len(self.name) == 1:
            return self.name[0]
        return str(self.name)

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other:CallbackName) -> bool: # type: ignore[override]
        return self.name == other.name
