import numpy as np
import math
import sys

from gplearn.functions import make_function

converter = {
    'sub': lambda x, y: x - y,
    'div': lambda x, y: x / y,
    'mydiv': lambda x, y: x / y,
    'mul': lambda x, y: x * y,
    'add': lambda x, y: x + y,
    'neg': lambda x: -x,
    'pow': lambda x, y: x ** y,
    'sin': lambda x: math.sin(x),
    'cos': lambda x: math.cos(x),
    'inv': lambda x: 1 / x,
    'inv_custom': lambda x: 1 / x,
    'sqrt': lambda x: abs(x) ** 0.5,
    'sqrt_custom': lambda x: x ** 0.5,
    'pow3': lambda x: x ** 3,
    'abs': lambda x: abs(x),
    #'sign': lambda x: -1 if x < 0 else 1 cannot use conditions here
}

threshold_func = sys.float_info.min

def _protected_division(x1, x2):
    """Closure of division (x1/x2) for zero denominator."""
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.array([np.divide(x1[i],x2[i]) if np.abs(x2[i]) > threshold_func else 1. for i in range(0,len(x1))])

def _sign(x): return np.sign(x)

#additional functions:
mydiv = make_function(function=_protected_division,
                    name='mydiv',
                    arity=2)
sign = make_function(function=_sign,
                    name='sign',
                    arity=1)