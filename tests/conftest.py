import numpy as np
import pytest


@pytest.fixture(scope="session")
def dummy_args():
    x = np.linspace(0, 1, 100)
    y_0 = 1

    return x, y_0
