from .constants import *
from types import SimpleNamespace

prefixes = SimpleNamespace(
    yotta=1e24,
    zetta=1e21,
    exa=1e18,
    peta=1e15,
    tera=1e12,
    giga=1e9,
    mega=1e6,
    kilo=1e3,
    hecto=1e2,
    deka=1e1,
    deci=1e-2,
    milli=1e-3,
    micro=1e-6,
    nano=1e-9,
    pico=1e-12,
    femto=1e-15,
    atto=1e-18,
    zepto=1e-21,
    yocto=1e-24,
)


def degrees_to_radians(degrees):
    return pi / 180 * degrees


def radians_to_degrees(radians):
    return 180 / pi * radians


def fahrenheit_to_celsius(fahrenheit):
    return (fahrenheit - 32) * 5 / 9


def celsius_to_fahrenheit(celsius):
    return (celsius * 9 / 5) + 32


def inches_to_metres(inches):
    return inches / 39.37


def metres_to_inches(metres):
    return metres * 39.37


def feet_to_metres(feet):
    return feet / 3.281


def metres_to_feet(metres):
    return metres * 3.281


def eV_to_joules(eV):
    return eV * q_e


def joules_to_eV(joules):
    return joules / q_e
