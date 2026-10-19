import pytest

from router.costs import PriceTable
from router.synthetic import SyntheticConfig, make_synthetic


@pytest.fixture(scope="session")
def syn():
    return make_synthetic(SyntheticConfig(n=1500, seed=7))


@pytest.fixture(scope="session")
def prices():
    return PriceTable.load()
