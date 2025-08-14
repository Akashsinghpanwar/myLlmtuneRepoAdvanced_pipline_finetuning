import marker
import importlib
import pkgutil

# Print all submodules
print("Marker package info:")
print(f"Version: {getattr(marker, '__version__', 'Not found')}")
print("\nAvailable submodules:")
for _, name, _ in pkgutil.iter_modules(marker.__path__, marker.__name__ + "."):
    print(f"- {name}")

# Try to find convert_file
print("\nSearching for convert_file...")
for _, name, _ in pkgutil.iter_modules(marker.__path__, marker.__name__ + "."):
    try:
        module = importlib.import_module(name)
        if hasattr(module, "convert_file"):
            print(f"Found convert_file in {name}")
            print(f"Usage: from {name} import convert_file")
    except ImportError:
        print(f"Could not import {name}")

# Try common patterns
print("\nTrying common import patterns:")
try:
    from marker.api import convert_file
    print("Success: from marker.api import convert_file")
except ImportError:
    print("Failed: from marker.api import convert_file")

try:
    from marker.convert import convert_file
    print("Success: from marker.convert import convert_file")
except ImportError:
    print("Failed: from marker.convert import convert_file")

try:
    from marker.core import convert_file
    print("Success: from marker.core import convert_file")
except ImportError:
    print("Failed: from marker.core import convert_file")

try:
    from marker.marker import convert_file
    print("Success: from marker.marker import convert_file")
except ImportError:
    print("Failed: from marker.marker import convert_file")

# Check if there's a main function or class
print("\nChecking for main class:")
if hasattr(marker, "Marker"):
    print("Found marker.Marker class")
    m = marker.Marker()
    print(f"Methods: {[method for method in dir(m) if not method.startswith('_')]}")
