"""Weather-to-PV forecasting, with lazy aliases for historical model artifacts."""

from importlib import import_module
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
from pathlib import Path
import sys

_GROUPS = ('data', 'models', 'training', 'inference', 'evaluation', 'plotting', 'studies', 'online')
_LEGACY_MODULES = {
    path.stem: group
    for group in _GROUPS
    for path in (Path(__file__).parent / group).glob('pv*.py')
}


class _LegacyLoader(Loader):
    """Return the canonical module object so old pickles keep class identity."""

    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return import_module(self.target)

    def exec_module(self, module):
        pass


class _LegacyFinder(MetaPathFinder):
    """Resolve only former flat forecasting module names, without eager imports."""

    def find_spec(self, fullname, path=None, target=None):
        prefix = __name__ + '.'
        if not fullname.startswith(prefix):
            return None
        short = fullname[len(prefix):]
        if '.' in short or short not in _LEGACY_MODULES:
            return None
        return spec_from_loader(fullname, _LegacyLoader(f'{prefix}{_LEGACY_MODULES[short]}.{short}'))


if not any(isinstance(finder, _LegacyFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _LegacyFinder())
