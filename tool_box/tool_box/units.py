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


def kelvin_to_celsius(kelvin):
    return kelvin - 273.15


def celsius_to_kelvin(celsius):
    return celsius + 273.15


def kelvin_to_fahrenheit(kelvin):
    return celsius_to_fahrenheit(kelvin - 273.15)


def fahrenheit_to_kelvin(fahrenheit):
    return fahrenheit_to_celsius(fahrenheit) + 273.15


def inches_to_metres(inches):
    return inches / 39.37


def metres_to_inches(metres):
    return metres * 39.37


def feet_to_metres(feet):
    return feet / 3.281


def metres_to_feet(metres):
    return metres * 3.281


def miles_to_metres(miles):
    return miles * 1609.344


def metres_to_miles(metres):
    return metres / 1609.344


def parsecs_to_metres(parsecs):
    return parsecs * 3.0857e16


def metres_to_parsecs(metres):
    return metres / 3.0857e16


def au_to_metres(au):
    return au * 149597870700


def metres_to_au(metres):
    return metres / 149597870700


def eV_to_joules(eV):
    return eV * q_e


def joules_to_eV(joules):
    return joules / q_e


def lbs_to_grams(lbs):
    return lbs * 453.59237


def grams_to_lbs(grams):
    return grams / 453.59237


def lbs_to_kg(lbs):
    return lbs * 0.45359237


def kg_to_lbs(grams):
    return grams / 0.45359237


def tonnes_to_grams(tonne):
    return tonne * 1e6


def grams_to_tonnes(grams):
    return grams / 1e6


def tonnes_to_kg(tonne):
    return tonne * 1000


def kg_to_tonnes(kg):
    return kg / 1000


def ounces_to_grams(ounces):
    return ounces * 28.349523125


def grams_to_ounces(grams):
    return grams / 28.349523125


def ounces_to_kg(ounces):
    return ounces / 35.27396195


def kg_to_ounces(kg):
    return kg * 35.27396195


def stones_to_grams(stones):
    return stones * 6350.29


def grams_to_stones(kg):
    return kg / 6350.29


def stones_to_kg(stones):
    return stones * 6.35029


def kg_to_stones(kg):
    return kg / 6.35029


def minutes_to_seconds(minutes):
    return minutes * 60


def seconds_to_minutes(seconds):
    return seconds / 60


def hours_to_seconds(hours):
    return hours * 3600


def seconds_to_hours(seconds):
    return seconds / 3600


def days_to_seconds(days):
    return days * 86400


def seconds_to_days(seconds):
    return seconds / 86400


def weeks_to_seconds(weeks):
    return weeks * 604800


def seconds_to_weeks(weeks):
    return weeks / 604800


def years_to_seconds(years):
    return years * 31557600


def seconds_to_years(seconds):
    return seconds / 31557600
