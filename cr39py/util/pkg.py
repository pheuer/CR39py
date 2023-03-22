import inspect
import importlib

from cr39py import _root_module


__all__ = [
    "find_classes",
    "get_class",
]


def find_classes(module, all_classes=None):
    """
    Get a list of (name, obj) tuples for every class in a given module.

    Built to be called recursively through the package to get all classes
    """

    if all_classes is None:
        all_classes = set()

    for name in module.__all__:
        # Item is either a class or a module...
        try:
            submodule = importlib.import_module(f"{module.__name__}.{name}")

            # Add any classes from this module to the class list
            classes = inspect.getmembers(submodule, inspect.isclass)
            for name, obj in classes:
                all_classes.add((name, obj))

            all_classes = find_classes(submodule, all_classes=all_classes)

        except ModuleNotFoundError:
            pass

    return all_classes


def get_class(name):
    # Get a list of all classes defined in this package
    _classes = find_classes(_root_module)

    matches = [x for x in _classes if x[0].lower() == name.lower()]

    if len(matches) > 0:
        # Return the class object
        return matches[0][1]
    else:
        print("List of classes found:\n")
        print(sorted([x[0] for x in _classes]))
        raise ValueError(f"No class found by name: {name}")


if __name__ == "__main__":

    import panoptes.detector.stack as mod

    print(find_classes(mod))
