import importlib

import pytest

SUBPACKAGES = [
    "core",
    "sfm",
    "render",
    "perception",
    "generative",
    "neural3d",
    "studio",
]


def test_import_main():
    mod = importlib.import_module("product_holodex")
    assert mod is not None


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_import_subpackages(name):
    mod = importlib.import_module(f"product_holodex.{name}")
    assert mod is not None
