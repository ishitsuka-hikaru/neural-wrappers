import numpy as np
from typing import Union, _GenericAlias
from collections import OrderedDict

Number = Union[float, int, np.number, np.float64]
Dict = Union[dict, OrderedDict]
Sequence = Union[list, tuple, set, np.ndarray]

# @brief Returns true if whatType is subclass of baseType. The parameters can be instantiated objects or types. In the
#  first case, the parameters are converted to their type and then the check is done.
def isBaseOf(whatType, baseType):
	if type(whatType) != type:
		whatType = type(whatType)
	if type(baseType) != type:
		baseType = type(baseType)
	return baseType in type(object).mro(whatType)

# Given a Type and a dictionary of {Type : Item}, returns the first Item that matches any ancestor of Type (assumed in
#  order of importance!)
# Example: B (extends) A (extends) Base (extends) Object
# pickTypeFromMRO(B, {Base: "msg1", A: "msg2", Object: "msg3"}) will return msg2 because that's how mro() works.
def pickTypeFromMRO(Type, switchType):
	Type = type(Type) if type(Type) != type else Type
	typeMRO = Type.mro()
	for Type in typeMRO:
		if Type in switchType:
			return switchType[Type]
	assert False, "%s not in %s" % (typeMRO, switchType)

# @brief Returns true if the item is of that particular type. Can be used for complex types (Number, Dict etc.) as
#  well, by checking __args__, which works on Unions.
# @param[in] item The item whose type is checked
# @param[in] Type The type 
def isType(item, Type : Union[type, _GenericAlias]) -> bool:
	itemType = type(item) if type(item) != type else item
	if hasattr("__args__", Type):
		return itemType in Type.__args__
	else:
		return itemType is Type