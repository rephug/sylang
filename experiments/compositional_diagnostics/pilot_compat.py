"""Read-only access to the pilot's and the copy probe's modules on a CPU machine without torch.

The pilot's ``data.py`` imports ``study3`` -> ``language`` -> ``followup`` (torch) and ``study3`` -> ``tasks``
-> ``model`` (torch). None of the functions used here (``study3.split_of``, ``data.CaseStream``,
``data.generate_cases``, ``data.select_answers``, ``data.lookup_answers``, ``data.group_by_subject``,
``sylang_batch``) touches torch, so when torch is not installed this module registers three empty
placeholder modules (``torch``, ``followup``, ``model``) before importing. Nothing in either directory is
modified. When torch *is* installed the real modules are imported instead.

Fidelity check: ``analyze_existing.py`` regenerates the frozen v1 dev and test case files through this path and
compares them with the SHA-256 ``split_hashes`` in ``pretrained_pilot/data_v1_manifest.json``.
"""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
COPY_PROBE = EXPERIMENTS / 'copy_probe'
PILOT = EXPERIMENTS / 'pretrained_pilot'
V1_MANIFEST = PILOT / 'data_v1_manifest.json'

STUBBED: list[str] = []


def _ensure_paths() -> None:
    for path in (str(COPY_PROBE), str(PILOT)):
        if path not in sys.path:
            sys.path.append(path)  # appended: this directory's own modules (grader.py, ...) keep precedence


def _stub_torch() -> None:
    if 'torch' in sys.modules or importlib.util.find_spec('torch') is not None:
        return
    for name in ('torch', 'followup', 'model'):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
            STUBBED.append(name)
    sys.modules['followup'].Runtime = None
    for constant in ('BOS', 'EOS', 'IGNORE', 'PAD'):
        setattr(sys.modules['model'], constant, 0)


def sylang_batch():
    """The public Sylang codec (header-free v2 codes); has no torch dependency."""
    _ensure_paths()
    import sylang_batch as module
    return module


_STUDY3 = []


def study3():
    if _STUDY3:
        return _STUDY3[0]
    _ensure_paths()
    _stub_torch()
    import study3 as module
    _STUDY3.append(module)
    return module


def pilot_data():
    """The pilot's ``data.py`` (imported under the name ``data``; read-only use)."""
    _ensure_paths()
    _stub_torch()
    # this directory has no module called ``data``; import the pilot's explicitly by path
    if 'pilot_data_module' in sys.modules:
        return sys.modules['pilot_data_module']
    spec = importlib.util.spec_from_file_location('pilot_data_module', PILOT / 'data.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['pilot_data_module'] = module
    spec.loader.exec_module(module)
    return module


def pilot_grader():
    _ensure_paths()
    spec = importlib.util.spec_from_file_location('pilot_grader_module', PILOT / 'grader.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
