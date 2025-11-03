from .constants import pi

__all__ = [
    "Circle"
]

class Circle:
    def __init__(self, radius):
        self.radius = radius
        self.area = pi * (radius**2)
        self.circumference = 2 * pi * radius
