from .base import FeatureExtractor

import inspect
import glob
import os
import importlib


def _is_abstract_extractor(extractor_cls):
    return extractor_cls.type == 'base'


def iter_extractor_clss():
    """Iterate over all of the extractors that are included in this sub-package.
    This is a convenience method for capturing all new extractors that are added
    over time and it is used both by the unit tests and in the
    ``extractor.__init__`` method.
    """
    return iter_subclasses(
        os.path.dirname(os.path.abspath(__file__)),
        FeatureExtractor,
        _is_abstract_extractor,
    )


def _iter_package_module_names(package_root):
    init_filename = os.path.join(package_root, '__init__.py')
    for py_filename in sorted(glob.glob(os.path.join(package_root, '*.py'))):
        if py_filename != init_filename:
            filename_root, _ = os.path.splitext(py_filename)
            module_name = os.path.basename(filename_root)
            yield module_name


def _iter_module_subclasses(package, module_name, base_cls):
    """inspect all modules in this directory for subclasses of inherit from
    ``base_cls``. inpiration from http://stackoverflow.com/q/1796180/564709
    """
    module = importlib.import_module('.' + module_name, package)
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, base_cls):
            yield obj


def iter_subclasses(package_root, base_cls, is_abstract):
    """Iterate over all instances of ``extractor`` subclasses.
    """
    package = os.path.basename(package_root)
    for module_name in _iter_package_module_names(package_root):
        for cls in _iter_module_subclasses(package, module_name, base_cls):
            if not is_abstract(cls):
                yield (cls.type, cls)

