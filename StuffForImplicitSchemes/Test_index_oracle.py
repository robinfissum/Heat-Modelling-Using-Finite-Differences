import unittest
from StuffForImplicitSchemes.index_oracle import IndexOracle


class TestIndexOracle(unittest.TestCase):
    def test_initialization(self):
        with self.assertRaises(ValueError):
            oracle = IndexOracle(-10, 20, 20)
        with self.assertRaises(ValueError):
            oracle = IndexOracle(10, 0, 7)

    def test_dimensions(self):
        oracle = IndexOracle(2, 2, 2)
        self.assertEqual(oracle._x_num, 2, msg='Should be 2')
        self.assertEqual(oracle._y_num, 2, msg='Should be 2')
        self.assertEqual(oracle._z_num, 2, msg='Should be 2')
        self.assertEqual(oracle._shape, (2, 2, 2), msg='Should be (2,2,2)')
        self.assertEqual(oracle._size, 8, msg='Should be 8')
        self.assertEqual(oracle._layer_size, 4, msg='Should be 4')

        oracle = IndexOracle(20, 30, 40)
        self.assertEqual(oracle._x_num, 20, msg='Should be 20')
        self.assertEqual(oracle._y_num, 30, msg='Should be 30')
        self.assertEqual(oracle._z_num, 40, msg='Should be 40')
        self.assertEqual(oracle._shape, (20, 30, 40), msg='Should be (20,30,40)')
        self.assertEqual(oracle._size, 20*30*40, msg='Should be 24000')
        self.assertEqual(oracle._layer_size, 20*30, msg='Should be 600')

    def test_to_3d(self):
        oracle = IndexOracle(2, 2, 2)
        self.assertEqual(oracle.to_3d(0), (0, 0, 0))
        self.assertEqual(oracle.to_3d(1), (1, 0, 0))
        self.assertEqual(oracle.to_3d(2), (0, 1, 0))
        self.assertEqual(oracle.to_3d(3), (1, 1, 0))
        self.assertEqual(oracle.to_3d(4), (0, 0, 1))
        self.assertEqual(oracle.to_3d(5), (1, 0, 1))
        self.assertEqual(oracle.to_3d(6), (0, 1, 1))
        self.assertEqual(oracle.to_3d(7), (1, 1, 1))

        oracle = IndexOracle(4, 3, 3)
        expected = [(0,0,0), (1,0,0), (2,0,0), (3,0,0), (0,1,0), (1,1,0), (2,1,0), (3,1,0), (0,2,0), (1,2,0), (2,2,0), (3,2,0),
                  (0,0,1), (1,0,1), (2,0,1), (3,0,1), (0,1,1), (1,1,1), (2,1,1), (3,1,1), (0,2,1), (1,2,1), (2,2,1), (3,2,1),
                  (0,0,2), (1,0,2), (2,0,2), (3,0,2), (0,1,2), (1,1,2), (2,1,2), (3,1,2), (0,2,2), (1,2,2), (2,2,2), (3,2,2)]
        for n in range(36):
            self.assertEqual(oracle.to_3d(n), expected[n])

    def test_to_1d(self):
        oracle = IndexOracle(2, 2, 2)
        self.assertEqual(oracle.to_1d(0, 0, 0), 0)
        self.assertEqual(oracle.to_1d(1, 0, 0), 1)
        self.assertEqual(oracle.to_1d(0, 1, 0), 2)
        self.assertEqual(oracle.to_1d(1, 1, 0), 3)
        self.assertEqual(oracle.to_1d(0, 0, 1), 4)
        self.assertEqual(oracle.to_1d(1, 0, 1), 5)
        self.assertEqual(oracle.to_1d(0, 1, 1), 6)
        self.assertEqual(oracle.to_1d(1, 1, 1), 7)

        oracle = IndexOracle(4, 3, 3)
        expected = [(0,0,0), (1,0,0), (2,0,0), (3,0,0), (0,1,0), (1,1,0), (2,1,0), (3,1,0), (0,2,0), (1,2,0), (2,2,0), (3,2,0),
                  (0,0,1), (1,0,1), (2,0,1), (3,0,1), (0,1,1), (1,1,1), (2,1,1), (3,1,1), (0,2,1), (1,2,1), (2,2,1), (3,2,1),
                  (0,0,2), (1,0,2), (2,0,2), (3,0,2), (0,1,2), (1,1,2), (2,1,2), (3,1,2), (0,2,2), (1,2,2), (2,2,2), (3,2,2)]
        for n in range(36):
            self.assertEqual(oracle.to_1d(*expected[n]), n)





'''
# Tests 1
A = IndexOracle(4, 4, 4)
print(f'X={A.to_1d(0,0,0)}')
print(f'Y={A.to_3d(0)}')
print(f'X={A.to_1d(1,0,0)}')
print(f'Y={A.to_3d(1)}')
print(f'X={A.to_1d(3,3,3)}')
print(f'Y={A.to_3d(63)}')
# Should raise error:
print(f'X={A.to_1d(3,3,4)}')

# Test 2
B = IndexOracle(100, 100, 100)
print(B.to_1d(38, 23, 90))
print(B.to_3d(902338))
# Should raise ValueError:
B = IndexOracle(-10, 20, 20)

# Test 3
B = IndexOracle(20, 30, 40)
print(B.to_1d(0, 0, 0))
print(B.to_1d(19, 29, 39))
print(B.to_3d(23999))
# Should raise error:
print(B.to_1d(22, 50, 100))

# Test 4:
# Should raise ValueError
A = IndexOracle(10, 0, 7)
'''