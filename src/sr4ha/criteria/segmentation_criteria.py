
def increase(score, score_prev, saturation = 1e-10, factor = 1):
    return score > saturation or factor * score >= score_prev

def decrease(error, error_prev, saturation = 1e-10, factor = 1):
    return error < saturation or factor * error <= error_prev