import emd_falsify as emd
import numpy as np
from scityping import Serializable

def test_interp1d():
    """Test serialization of interp1d"""

    rng = np.random.RandomState(569465)
    ppf = emd.make_empirical_risk_ppf(rng.uniform(size=100))

    ppf2 = Serializable.validate(Serializable.deep_reduce(ppf))

    Φarr = np.linspace(0, 1)
    assert np.array_equal(ppf(Φarr), ppf2(Φarr))
