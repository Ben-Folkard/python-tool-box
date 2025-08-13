from tool_box.constants import pi
from tool_box.units import prefixes, degrees_to_radians
from tool_box.geometry import Circle
import tool_box.nuclear as nc

print(degrees_to_radians(180))
print(pi)
print(prefixes.yotta)
print(Circle(radius=1).area)
print(nc.half_life_decay(initial_amount=100, half_life=4, time_passed=8))
print(nc.decay_constant_decay(initial_amount=100, decay_constant=nc.half_life_to_decay_constant(4), time_passed=8))
print(nc.neutron_wavelength_to_energy(nc.neutron_energy_to_wavelength(100)))
