import numpy as np
from colour.core import lab_to_lch

def test_lab_to_lch():
    """Test the lab_to_lch conversion function."""
    L, a, b = 50, 10, 10
    L_out, C, h = lab_to_lch(L, a, b)
    
    assert L_out == L
    assert np.isclose(C, 14.142, atol=0.01)
    assert np.isclose(h, 45.0, atol=0.01)
