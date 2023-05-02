import os

import numpy as np

from proj5_code.classification.stats_helper import compute_mean_and_std


def test_mean_and_variance():
    if os.path.exists("proj5_code/proj5_unit_tests/classification/small_data/"):
        mean, std = compute_mean_and_std("proj5_code/proj5_unit_tests/classification/small_data/")
    else:
        mean, std = compute_mean_and_std("../proj5_code/proj5_unit_tests/classification/small_data/")
    assert np.allclose(mean, np.array([0.46178914]))
    assert np.allclose(std, np.array([0.256041]))

