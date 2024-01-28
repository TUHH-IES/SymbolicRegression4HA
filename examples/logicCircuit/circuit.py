def subtractor(x: bool, y: bool, Bin: bool):
    a = x ^ y
    d = a ^ Bin
    b = not x
    c = y and b
    e = not a
    f = e and Bin
    Bout = c or f
    return [d, Bout]