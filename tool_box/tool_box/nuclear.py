from .constants import ln, e, h, c, m_n, sqrt

# Linear decay chains and the Bateman equation: https://bjodah.github.io/posts/bateman-equation.html


def decay_constant_to_half_life(decay_constant):
    return ln(2) / decay_constant


def half_life_to_decay_constant(half_life):
    return ln(2) / half_life


def neutron_energy_to_wavelength(energy):
    return h * c / sqrt(energy**2 - (m_n * c**2)**2)


def neutron_wavelength_to_energy(wavelength):
    return sqrt((m_n * c**2)**2 + (h / wavelength * c)**2)


def half_life_decay(initial_amount, half_life, time_passed):
    """
    Calculates the amount a thing would decay in a given amount of time based on a given half life
    """
    return initial_amount * (0.5)**(time_passed / half_life)


def decay_constant_decay(initial_amount, decay_constant, time_passed):
    """
    Calculates the amount a thing would decay in a given amount of time based on a given decay constant
    """
    return initial_amount * e**(-decay_constant*time_passed)
