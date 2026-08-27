"""Model auto-discovery.

Scans every module in fii/models for BaseModel subclasses and registers
them by their ``name``.  Dropping a new file into this folder is the
ONLY step needed to make ``python pipeline.py --model <name>`` work.
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil

import fii.models
from fii.models.base import BaseModel


def _discover() -> dict[str, type[BaseModel]]:
    found: dict[str, type[BaseModel]] = {}
    for info in pkgutil.iter_modules(fii.models.__path__):
        if info.name.startswith("_") or info.name in ("base", "registry"):
            continue
        mod = importlib.import_module(f"fii.models.{info.name}")
        for _, cls in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(cls, BaseModel) and cls is not BaseModel
                    and cls.name):
                found[cls.name] = cls
    return found


def available() -> list[str]:
    return sorted(_discover())


def get_model(name: str) -> BaseModel:
    models = _discover()
    if name not in models:
        raise KeyError(f"unknown model '{name}'; available: "
                       f"{', '.join(sorted(models))}")
    return models[name]()
