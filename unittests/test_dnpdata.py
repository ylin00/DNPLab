import unittest

import numpy as np
from numpy.testing import assert_array_equal

from dnplab.dnpData import dnpdata

from .testing import get_gauss_3d


class dnpDataTester(unittest.TestCase):
    def setUp(self):
        self.x, self.y, self.z, self.gauss_3d = get_gauss_3d(0.1)
        self.dnpdata = dnpdata(self.gauss_3d, [self.x, self.y, self.z], ["x", "y", "z"])

    def test_dnpdata(self):
        assert_array_equal(self.dnpdata.coords["x"], self.x)
        assert_array_equal(self.dnpdata.dims, ["x", "y", "z"])

    def test_dnpdata_get(self):
        assert_array_equal(self.dnpdata["x", 0:10], self.gauss_3d[0:10, :, :])
        assert_array_equal(self.dnpdata["y", 0:10], self.gauss_3d[:, 0:10, :])

    def test_coords_get_set(self):
        self.dnpdata.new_dim("r", np.r_[0:10])
        assert_array_equal(self.dnpdata.dims, ["x", "y", "z", "r"])
        self.dnpdata.rename("r", "s")
        assert_array_equal(self.dnpdata.dims, ["x", "y", "z", "s"])

    def test_coords_sort_reorder(self):
        self.dnpdata.reorder(["y", "z", "x"])
        assert_array_equal(self.dnpdata.dims, ["y", "z", "x"])
        assert_array_equal(self.dnpdata.coords["z"], self.z)
        self.dnpdata.sort_dims()
        assert_array_equal(self.dnpdata.dims, ["x", "y", "z"])
        assert_array_equal(self.dnpdata.coords["z"], self.z)

    def test_squeeze(self):
        np.random.seed(0)
        a = np.random.rand(20, 10, 10, 1, 1)
        b = np.random.rand(20, 1, 10, 1, 20)
        c = np.random.rand(1, 1, 20, 10)
        a = dnpdata(a)
        b = dnpdata(b)
        c = dnpdata(c)
        assert_array_equal(a.squeeze(), np.reshape(a, (20, 10, 10)))
        assert_array_equal(b.squeeze(), np.reshape(b, (20, 10, 20)))
        assert_array_equal(c.squeeze(), np.reshape(c, (20, 10)))

        # Squeezing to 0-dim should still give an dnpdata
        a = [[[1.5]]]
        a = dnpdata(a)
        res = a.squeeze()
        self.assertEqual(res.values, 1.5)
        self.assertEqual(res.ndim, 0)
        self.assertEqual(type(res), dnpdata)


if __name__ == "__main__":
    pass
