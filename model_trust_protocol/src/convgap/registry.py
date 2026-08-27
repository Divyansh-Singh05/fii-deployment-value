"""Application registration and discovery."""

from __future__ import annotations

import importlib
import pkgutil

from convgap.application import Application, Layer
from convgap.friction import FrictionLevel
from convgap.role import Role

_REGISTRY: dict[str, type[Application]] = {}


class DuplicateApplicationKeyError(ValueError):
    """Raised when two applications claim the same key."""


def register(cls: type[Application]) -> type[Application]:
    """Class decorator adding an application to the registry."""
    if cls.key in _REGISTRY:
        raise DuplicateApplicationKeyError(f"key {cls.key!r} already registered")
    _REGISTRY[cls.key] = cls
    return cls


def discover() -> None:
    """Import every module under :mod:`convgap.applications` so decorators run."""
    import convgap.applications as pkg

    for info in pkgutil.iter_modules(pkg.__path__):
        importlib.import_module(f"{pkg.__name__}.{info.name}")


def all_applications() -> list[Application]:
    """Instantiate every registered application, ordered by key."""
    discover()
    return [_REGISTRY[k]() for k in sorted(_REGISTRY)]


def controls() -> list[Application]:
    """Entries drawn from outside the dataset under study.

    A control failing at the same level as the applications indicates the
    friction is a property of the evaluation transition rather than of this
    dataset; a control that clears it indicates the opposite.
    """
    return [a for a in all_applications() if a.role is Role.CONTROL]


def survivors() -> list[Application]:
    """Applications of the dataset that clear every level of the ladder."""
    return [a for a in all_applications() if a.fails_at is None and a.role is Role.APPLICATION]


def by_friction() -> list[tuple[FrictionLevel | None, list[Application]]]:
    """Group by the level at which each application dies, ascending.

    This is the ordering of the paper's central exhibit. Where each application
    stops is the finding; read across them, the ladder maps the boundary of the
    dataset's usefulness.
    """
    apps = all_applications()
    out: list[tuple[FrictionLevel | None, list[Application]]] = []
    for level in FrictionLevel:
        matching = [a for a in apps if a.fails_at is level]
        if matching:
            out.append((level, matching))
    clearing = [a for a in apps if a.fails_at is None]
    if clearing:
        out.append((None, clearing))
    return out


def by_instrument() -> list[tuple[Layer, list[Application]]]:
    """Group by the stress-test environment used, in declaration order."""
    apps = all_applications()
    return [
        (layer, matching) for layer in Layer if (matching := [a for a in apps if a.layer is layer])
    ]


def coverage_gaps() -> list[Layer]:
    """Instruments with no registered application.

    The dataset is claimed to have been stressed across the modelling stack.
    An unused instrument is a hole in that claim and is reported, not ignored.
    """
    covered = {a.layer for a in all_applications()}
    return [layer for layer in Layer if layer not in covered]
