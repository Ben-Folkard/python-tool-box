# Initializing all the submodules
# (weird behaviour otherwise)
"""
from . import constants
from . import geometry
from . import nuclear
from . import units
from . import plotter
"""

from os.path import dirname, join, basename
from glob import glob
from importlib import import_module

# Get __init__.py file location
package_dir = dirname(__file__)

modules = []

for py_file in glob(join(package_dir, "*.py")):
    module_name = basename(py_file)[:-3]
    if module_name == "__init__":
        continue
    import_module(f".{module_name}", package=__name__)
    modules.append(module_name)

__all__ = modules