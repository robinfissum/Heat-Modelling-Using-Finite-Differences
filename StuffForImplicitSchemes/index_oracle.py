
class IndexOracle:
    """
        For the implicit method!
        For converting between 3-dimensional and 1-dimensional indices.
        Specifically, if A is a 3 dimensional-matrix (a tensor) with elements indexed as
        A[i,j,k] for 0<=i<=x_num-1, 0<=j<=y_num-1, 0<=k<=z_num-1, then the matrix A
        can be vectorized as a vector B of length x_num*y_num*z_num indexed as
        B[s] for 0<=s<=x_num*y_num*z_num-1
        Through testing, it is clear that computing indices on the fly
        between 1d and 3d is more efficient than storing the indices in a dictionary
        (at least for reasonably sized index sets).

        Usage:
        >>> foo = IndexOracle(5, 7, 10)
        >>> foo.to_1d(2, 3, 6)
        227
        >>> foo.to_3d(227)
        (2, 3, 6)
    """

    def __init__(self, x_num: int, y_num: int, z_num: int):
        if x_num <= 0 or y_num <= 0 or z_num <= 0:
            raise ValueError(
                f'InvalidDimensions {x_num},{y_num},{z_num} passed to IndexOracle. Positive integers were expected.')
        self._x_num = x_num
        self._y_num = y_num
        self._z_num = z_num
        self._shape = (x_num, y_num, z_num)
        self._size = x_num * y_num * z_num
        self._layer_size = x_num * y_num

    def __str__(self):
        return f'IndexOracle of size {(self._x_num,self._y_num,self._z_num)}<-->[{0},...,{self._size-1}].'

    def to_1d(self, *index_triple: int) -> int:
        if len(index_triple) != 3 or not all([i >= 0 for i in index_triple]):
            raise ValueError(f'IndexOracle.to_1d takes 3 non-negative integer arguments. Input was: {index_triple}')
        elif any([index_triple[i] >= self._shape[i] for i in range(3)]):
            raise ValueError(
                f'Index out of bounds in IndexOracle.to_1d. Valid indices are {{0,1,..,{self._shape[0] - 1}}}x{{0,1,..,{self._shape[1] - 1}}}x{{0,1,..,{self._shape[2] - 1}}}, but {index_triple} was passed.')
        return index_triple[2] * self._layer_size + index_triple[1] * self._x_num + index_triple[0]

    def to_3d(self, num: int) -> tuple:
        if not 0 <= num < self._size:
            raise ValueError(f'Index out of bounds for IndexOracle.to_3d. Expected integer in [{0},...,{self._size-1}], but input was: {num}')
        k = num // self._layer_size
        j = (num - k * self._layer_size) // self._x_num
        i = num - k * self._layer_size - j * self._x_num
        return i, j, k
