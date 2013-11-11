import unittest
from context import lily
import numpy as np

class TestLily(unittest.TestCase):
  def test_classify_0(self):
    """
    lily - classify_0 returns the majority class as the prediction
    """
    large_shape = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['foo', 'bar', 'bz', 'quux']
    result = lily.classify_0(large_shape[1, :], large_shape[2, :], labels, 2)
    assert result == 'foo'

if __name__ == '__main__':
    unittest.main()

