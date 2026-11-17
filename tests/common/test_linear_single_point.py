"""1-point linear model: real validator accepts it, 0 points rejected."""

import pytest

from cyclescope.tree import validate


AXES = {"transfer_bytes": {"kind": "int", "description": "op bytes"}}


def test_single_point_linear_accepted():
    validate._check_model(
        {"kind": "linear", "axis": "transfer_bytes",
         "points": [[512, 33.6]]}, AXES, label="t")


def test_zero_point_linear_rejected():
    with pytest.raises(Exception):
        validate._check_model(
            {"kind": "linear", "axis": "transfer_bytes", "points": []},
            AXES, label="t")


def test_two_point_linear_still_accepted():
    validate._check_model(
        {"kind": "linear", "axis": "transfer_bytes",
         "points": [[512, 33.6], [1024, 40.0]]}, AXES, label="t")
