import os
import sys
import numpy as np
import gymnasium as gym
import pytest

# Make project root importable when running under pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from feature_extractors import RBFFeatureExtractor


def test_rbf_interface_and_normalize_state():
    """Starter checks for RBFFeatureExtractor interface and normalize_state."""
    env = gym.make('MountainCar-v0')
    rbf = RBFFeatureExtractor(env, n_centers=9, sigma=0.1)

    # Basic attributes
    assert hasattr(rbf, 'n_centers') and isinstance(rbf.n_centers, int)
    assert hasattr(rbf, 'sigma') and isinstance(rbf.sigma, float)
    assert hasattr(rbf, 'n_features') and isinstance(rbf.n_features, int)

    # centers attribute shape
    assert hasattr(rbf, 'centers')
    assert isinstance(rbf.centers, np.ndarray)
    assert rbf.centers.shape[0] == rbf.n_centers
    assert rbf.centers.shape[1] == 2

    # normalize_state behavior
    low = np.array([rbf.position_bounds[0], rbf.velocity_bounds[0]])
    high = np.array([rbf.position_bounds[1], rbf.velocity_bounds[1]])
    mid = (low + high) / 2.0

    assert np.allclose(rbf.normalize_state(low), np.array([0.0, 0.0]), atol=1e-8)
    assert np.allclose(rbf.normalize_state(high), np.array([1.0, 1.0]), atol=1e-8)
    assert np.allclose(rbf.normalize_state(mid), np.array([0.5, 0.5]), atol=1e-8)


def test_extract_features_contract_xfail():
    """Check that extract_features exists and xfail if not implemented yet.

    If students have implemented `extract_features`, this test will exercise
    the basic contract (return type and length). Otherwise it is an expected
    xfail to indicate the student task is incomplete.
    """
    env = gym.make('MountainCar-v0')
    rbf = RBFFeatureExtractor(env, n_centers=9, sigma=0.1)

    assert hasattr(rbf, 'extract_features') and callable(rbf.extract_features)

    state = np.array([rbf.position_bounds[0], rbf.velocity_bounds[0]])

    try:
        feats = rbf.extract_features(state)
    except NotImplementedError:
        pytest.xfail("extract_features not implemented yet (student exercise)")

    assert isinstance(feats, np.ndarray)
    assert feats.shape == (rbf.n_features,)
    assert np.all(feats >= 0) and np.all(feats <= 1)



