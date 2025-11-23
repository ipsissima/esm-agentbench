
import math

def normalize_vector(vec):
    total = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / total for v in vec]
