import numpy as np
import scipy.linalg as la

class DIIS(object):
  """DIIS extrapolation class.

  This uses the CROP algorithm, which I found in a `JCTC article`_.

  Attributes:
    options (dict): Options controlling the DIIS extrapolation, such as the
      minimum and maximum number of entries to use.
    arrays_list (list of tuple of array-like objects): The variables to be
      extrapolated.
    errors_list (list of tuple of array-like objects): The residuals of the
      variables to be extrapolated.

  .. _JCTC article:
    http://doi.org/10.1021/ct501114q
  """

  _option_defaults = {
    'n_max': 3,
    'n_min': 3
  }

  def __init__(self, **options):
    """Initialize DIIS object.

    Args:
      n_max (:obj:`int`, optional): Maximum number of vectors to store.
      n_min (:obj:`int`, optional): Minimum number of vectors to store.
    """
    self.options = DIIS._option_defaults
    self.options.update(options)
    self.arrays_list = []
    self.errors_list = []

  def add_entry(self, *array_error_pairs):
    """Add an entry to the DIIS extrapolation.

    Drops entries to make room and then appends the new entry.

    Args:
      *array_error_pairs (list of tuples): Each tuple is a pair of lists of
        array-like object.  The first one contains the arrays to be extrapolated
        and the second one contains the corresponding residuals.
    """
    while(len(self.arrays_list) > self.options['n_max'] - 1):
      self.arrays_list.pop(0)
      self.errors_list.pop(0)
    arrays, errors = zip(*array_error_pairs)
    self.arrays_list.append(arrays)
    self.errors_list.append(errors)

  def extrapolate(self):
    """Compute the DIIS extrapolation.

    Returns:
      tuple: The last element is the norm of the residual.  The remaining
        elements are the extrapolated arrays.
    """
    if len(self.arrays_list) < self.options['n_min']:
      return self.arrays_list[-1] + (1.0,)
    coeffs, error_norm = self._get_extrapolation_coefficients()
    arrays = tuple(sum(coeff * array for coeff, array in zip(coeffs, arrays))
                   for arrays in zip(*self.arrays_list))
    errors = tuple(sum(coeff * error for coeff, error in zip(coeffs, errors))
                   for errors in zip(*self.errors_list))
    self.arrays_list[-1] = arrays
    self.errors_list[-1] = errors
    return arrays + (error_norm,)

  def _get_extrapolation_coefficients(self):
    """Compute DIIS extrapolation coefficients.

    Follows the notation of Pulay's `Improved SCF Convergence Acceleration`_.

    .. _Improved SCF Convergence Acceleration:
      http://doi.org/10.1002/jcc.540030413
    """
    n = len(self.arrays_list)
    B = np.zeros((n, n))
    for i in range(n):
      for j in range(i, n):
        B[i, j] = sum(np.vdot(error1.view(np.ndarray), error2.view(np.ndarray))
                     for error1, error2 in
                     zip(self.errors_list[i], self.errors_list[j]))
        B[j, i] = B[i, j]

    # Build the matrix A = [[B, -1], [-1, 0]]
    A = - np.ones((n+1, n+1))
    A[:n, :n] = B
    A[n, n] = 0

    # Build the vector b = [0, ..., 0, -1]
    b = np.array([0.] * n + [-1.])

    # Solve A * x = b
    x = la.solve(A, b)

    # Return the extrapolation coefficients and the norm of the error
    coeffs, error_norm = x[:n], x[n]
    return coeffs, error_norm
