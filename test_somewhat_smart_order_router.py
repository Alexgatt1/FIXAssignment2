from __future__ import annotations

import math

import pytest

from somewhat_smart_order_router import best_price_improvement


def test_best_price_improvement_normal_order() -> None:
    """
    Normal case: a reasonable buy order with a typical NBBO.

    This test checks that:
    - the function returns an exchange as a string
    - the predicted price improvement is a finite float
    """
    try:
        best_exchange, predicted_improvement = best_price_improvement(
            symbol="AAPL",
            side="B",
            quantity=100,
            limit_price=190.00,
            bid_price=189.90,
            ask_price=190.10,
            bid_size=5_000,
            ask_size=6_000,
        )
    except RuntimeError:
        pytest.skip("No exchange models are loaded yet; train models before running this test.")

    assert isinstance(best_exchange, str)
    assert best_exchange != ""
    assert isinstance(predicted_improvement, float)
    assert math.isfinite(predicted_improvement)


def test_best_price_improvement_corner_case_zero_quantity() -> None:
    """
    Corner case: zero-quantity order.

    Even though a zero-quantity order is not realistic, the router should still:
    - return some exchange name
    - return a finite float for predicted price improvement
    (or raise RuntimeError if models are not yet available).
    """
    try:
        best_exchange, predicted_improvement = best_price_improvement(
            symbol="AAPL",
            side="S",      
            quantity=0,   
            limit_price=190.00,
            bid_price=189.90,
            ask_price=190.10,
            bid_size=5_000,
            ask_size=6_000,
        )
    except RuntimeError:
        pytest.skip("No exchange models are loaded yet; train models before running this test.")

    assert isinstance(best_exchange, str)
    assert best_exchange != ""
    assert isinstance(predicted_improvement, float)
    assert math.isfinite(predicted_improvement)
