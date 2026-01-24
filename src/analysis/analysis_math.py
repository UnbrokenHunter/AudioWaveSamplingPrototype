def lerp(a: float, b: float, t: float) -> float:
    """
    Linear interpolate on the scale given by a to b, using t as the point on that scale.
    """
    return (1 - t) * a + t * b
