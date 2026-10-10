# simple-chebyshev-polynomial

Evaluates Chebyshev polynomials of the first and second kind, T_n(x) and U_n(x), for any non-negative integer n and any real x.

```python
from simple_chebyshev_polynomial import chebyshev_t, chebyshev_u

chebyshev_t(5, 0.5)   # 0.5
chebyshev_u(5, 0.5)   # 0.0
chebyshev_t(10, 2.0)  # 262145.0
chebyshev_u(10, 2.0)  # 524287.0
```

## Why this exists

The library provides the bare minimum: two functions that evaluate a single Chebyshev polynomial at a single point. There is no vectorisation, no coefficient-array API, no root computation, no fitting routine. The intended use is as a building block in approximation and interpolation code where the caller already knows how to compose the recurrence.

The evaluation uses the three-term recurrence directly — T_{k+1}(x) = 2x T_k(x) - T_{k-1}(x), and the analogous formula for U — rather than the closed forms cos(n arccos x) and sin((n+1) arccos x)/sin(arccos x). The closed forms are only valid for |x| <= 1; the recurrence holds for all real x with no domain restrictions and avoids the transcendental-function call. The trade-off is that each call costs O(n) arithmetic; if you need T_n at many points you should compose your own loop.

## Edge cases

- `n` must be a non-negative integer. Passing a negative value raises `ValueError`; passing a float like `3.0` will work if it represents an integer but is not officially supported — pass `int`.
- For large `n` (say, > 1000) and |x| > 1, the magnitude of T_n and U_n grows exponentially and will overflow to `inf`. This is intrinsic to the polynomials, not a bug.
- The functions accept and return Python `float`. Integers passed as `x` are converted internally; the return type is always `float` (except T_0, which returns `1.0`).

## Running the tests

```
PYTHONPATH=src python -m unittest discover -s tests
```
