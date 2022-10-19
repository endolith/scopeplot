"""
Created on Sun May 24 2015

Density plot of waveforms
"""

from __future__ import division, print_function

import numpy as np
from numpy import asarray, linspace, arange, pad
import matplotlib.pyplot as plt
from scipy.signal import resample
from scipy.fftpack import next_fast_len

# minimum resampling factor
RS = 2


def _ceildiv(a, b):
    """Ceiling integer division"""
    return -(-a // b)


def _get_weights(N):
    """
    Sample weights for a chunk with N samples per chunk.  Determined by the
    amount a 1-pixel wide line centered on segment n would overlap the current
    pixel.

    So _get_weights(3) returns `[2/6, 4/6, 6/6, 4/6, 2/6]`, which means the
    center segment will only affect this pixel, the first segment's width will
    overlap this pixel by 4/6, the last segment of the previous chunk will
    overlap this pixel by 2/6, etc.
    """
    if abs(int(N)) != N or N == 0:
        raise ValueError("N must be a positive integer")

    den = 2*N

    if N % 2:  # odd
        num = np.concatenate((arange(2, 2*N + 1, 2), arange(2*N - 2, 0, -2)))
    else:  # even
        num = np.concatenate((arange(1, 2*N, 2), arange(2*N - 1, 0, -2)))

    return num/den


def _ihist(a, bins, range_):
    """
    interpolated histogram

    a is a sequence of samples, considered to be connected by lines, including
    overlap of neighboring pixels

    bins is number of bins to group them into vertically

    range_ is total range of output, which may be wider than the widest values
    in a
    """
    a = asarray(a)

    if a.ndim > 1:
        raise AttributeError('a must be a series. it will not be flattened')

    if (not np.isscalar(bins)) or (int(bins) != bins) or bins < 1:
        raise ValueError("`bins` should be a positive integer.")

    if a.size < 2 or a.size % 4 in {0, 3}:
        raise ValueError(f'not a valid size with overlap: {a.size}')

    mn, mx = [mi + 0.0 for mi in range_]  # Make float

    if a.min() < mn or a.max() > mx:
        raise NotImplementedError(
            f"values outside of range_ are not yet supported {a.min()} {a.max()}"
        )


    if (mn >= mx):
        raise AttributeError('max must be larger than '
                             'min in range_ parameter.')

    bin_edges = linspace(mn, mx, bins+1, endpoint=True)
    bin_width = (mx - mn)/bins

    pairs = np.vstack((a[:-1], a[1:])).T
    lower = np.minimum(pairs[:, 0], pairs[:, 1])
    upper = np.maximum(pairs[:, 0], pairs[:, 1])

    bin_lower = np.searchsorted(bin_edges, lower)
    bin_upper = np.searchsorted(bin_edges, upper)

    h = 1/(upper - lower)

    out = np.zeros(bins)

    weights = _get_weights(len(a) // 2)

    for n in range(len(pairs)):
        w = weights[n]
        lo = bin_lower[n]
        hi = bin_upper[n]
        if lo == hi:
            # Avoid divide by 0
            if lower[n] == bin_edges[lo]:
                # straddles 2 bins
                try:
                    out[lo-1] += w * 0.5
                    out[lo]   += w * 0.5
                except IndexError:
                    raise NotImplementedError('Values on edge of range_')
                    # TODO: Could handle this with more ifthens,
                    # but should be a smarter way
            else:
                out[lo-1] += w
        else:
            out[lo-1]    += w * h[n] * (bin_edges[lo] - lower[n])
            out[lo:hi-1] += w * h[n] * bin_width
            out[hi-1]    += w * h[n] * (upper[n] - bin_edges[hi-1])

    return out


def scopeplot(x, width=800, height=400, range_=None, cmap=None, plot=None):
    """
    Plot a signal using brightness to indicate density.

    Parameters
    ----------
    x : array_like, 1-D
        The signal to be plotted
    width, height : int, optional
        The width and height of the output image in pixels.  Default is
        800×400.
    range_ : float or 2-tuple of floats, optional
        The vertical range of the plot.  If a tuple, it is (xmin, xmax).  If
        a single number, the range is (-range, range).  If None, it autoscales.
    cmap : str or matplotlib.colors.LinearSegmentedColormap, optional
        A matplotlib colormap for
        Grayscale by default.
    plot : bool or str or None, optional
        If plot is None, the X image array is returned.
        if plot is True, the image is plotted directly.
        If plot is a string, it represents a filename to save the image to
        using matplotlib's `imsave`.

    Returns
    -------
    X : ndarray of shape (width, height)
        A 2D array of amplitude 0 to 1, representing the density of the signal
        at that point.

    """

    if cmap is None:
        cmap = 'gray'

    x = asarray(x)

    N = len(x)

    # Add zeros to end to reduce circular Gibbs effects
    MIN_PAD = 5   # TODO: what should this be?  Seems subjective.

    # Make input an optimal length for fast processing
    pad_amount = next_fast_len(N + MIN_PAD) - N

    x = pad(x, (0, pad_amount), 'constant')

    # Resample such that signal evenly divides into chunks of equal length
    new_size = int(round(_ceildiv(RS*N, width) * width / N * len(x)))
    print(f'new size: {new_size}')

    x = resample(x, new_size)

    if not range_:
        range_ = 1.1 * np.amax(np.abs(x))

    if np.size(range_) == 1:
        xmin, xmax = -range_, +range_
    elif np.size(range_) == 2:
        xmin, xmax = range_
    else:
        raise ValueError('range_ not understood')

    spp = _ceildiv(N * RS, width)  # samples per pixel
    norm = 1/spp

    # Pad some zeros at beginning for overlap
    x = pad(x, (spp//2, 0), 'constant')

    X = np.empty((width, height))

    chunksize = 2*spp if spp % 2 else 2*spp + 1
    print(f'spp: {spp}, chunk size: {chunksize}')

    for n in range(width):
        chunk = x[n*spp:n*spp+chunksize]
        assert len(chunk)  # don't send empties
        try:
            h = _ihist(chunk, bins=height, range_=(xmin, xmax))
        except ValueError:
            print('argh', len(chunk))
        else:
            X[n] = h * norm

    assert np.amax(X) <= 1.001, np.amax(X)

    X = X**(0.4)  # TODO: SUBJECTIVE

    if isinstance(plot, str):
        plt.imsave(plot, X.T, cmap=cmap, origin='lower', format='png')
    elif plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(X.T, origin='lower', aspect='auto', cmap=cmap,
                  extent=(0, len(x), xmin, xmax), interpolation='nearest')
    #              norm=LogNorm(vmin=0.01, vmax=300))
    else:
        return X


#########
# TESTS #
#########


def test_get_weights():
    from numpy.testing import assert_raises, assert_allclose

    assert_raises(ValueError, _get_weights, -3)
    assert_raises(ValueError, _get_weights, 0)

    assert_allclose(_get_weights(1), [1])
    assert_allclose(_get_weights(2), [1/4, 3/4, 3/4, 1/4])
    assert_allclose(_get_weights(3), [2/6, 4/6, 1, 4/6, 2/6])
    assert_allclose(_get_weights(4), [1/8, 3/8, 5/8, 7/8, 7/8, 5/8, 3/8, 1/8])
    assert_allclose(
        _get_weights(5),
        [2 / 10, 4 / 10, 6 / 10, 8 / 10, 1, 8 / 10, 6 / 10, 4 / 10, 2 / 10],
    )


    assert_allclose(np.sum(_get_weights(101)), 101)
    assert_allclose(np.sum(_get_weights(10111)), 10111)


def test_ihist(SLOW_TESTS=False):
    from numpy.testing import (assert_array_equal, assert_raises,
                               assert_allclose)

    # Invalid bins=
    assert_raises(ValueError, _ihist, [],     (10, 4), (0, 10))
    assert_raises(ValueError, _ihist, [1, 2], (10, 4), (0, 10))
    assert_raises(ValueError, _ihist, [],     -5, (0, 10))
    assert_raises(ValueError, _ihist, [1, 2], -5, (0, 10))
    assert_raises(ValueError, _ihist, [],     5.7, (0, 10))
    assert_raises(ValueError, _ihist, [1, 2], 5.7, (0, 10))

    # Incorrect number of samples with overlap
    """
    Because of overlap, valid number of samples fed to hist are
    2, 5, 6, 9, 10, 13, 14, 17, 18, 21, 22, 25, 26, 29, 30, 33, ...
    """
    assert_raises(ValueError, _ihist, [], 3, (0, 1))
    assert_raises(ValueError, _ihist, [1], 3, (0, 1))
    assert_raises(ValueError, _ihist, [0.5], 3, (0, 1))
    assert_raises(ValueError, _ihist, [3, 2, 1], 5, (0, 5))
    assert_raises(ValueError, _ihist, [0.5, 0.5, 0.5, 0.5], 3, (0, 1))
    assert_raises(ValueError, _ihist, np.ones(31), 5, (0, 5))

    # 1 sample per pixel = 1 segment per pixel
    assert_array_equal(_ihist([2, 3], 3, (0, 3)), [0, 0, 1])

    assert_allclose(_ihist([3.8, 5.8], 7, (3.5, 7)),
                    [0.1, 0.25, 0.25, 0.25, 0.15, 0, 0])

    assert_allclose(_ihist([5.8, 3.8], 7, (3.5, 7)),
                    [0.1, 0.25, 0.25, 0.25, 0.15, 0, 0])

    assert_allclose(_ihist([0, 1], 5, (0, 1)), [1/5, 1/5, 1/5, 1/5, 1/5])

    assert_array_equal(_ihist([5.5, 7], 10, (0, 10)),
                       [0, 0, 0, 0, 0, 1/3, 2/3, 0, 0, 0])

    assert_array_equal(_ihist([0, 1], 5, (0, 5)),     [1, 0, 0, 0, 0])
    assert_array_equal(_ihist([3, 2], 5, (0, 5)),     [0, 0, 1, 0, 0])
    assert_array_equal(_ihist([4, 5], 5, (0, 5)),     [0, 0, 0, 0, 1])
    assert_array_equal(_ihist([0.5, 2.5], 3, (0, 3)), [1/4, 1/2, 1/4])
    assert_array_equal(_ihist([0.5, 1.5], 3, (0, 3)), [1/2, 1/2, 0])

    # Single line appears in all one bin
    assert_array_equal(_ihist([0.5, 0.5], 3, (0, 1)), [0, 1, 0])
    assert_array_equal(_ihist([0.9, 0.9], 5, (0, 1)), [0, 0, 0, 0, 1])

    # Falls entirely on bin edge, so half in each neighboring bin
    assert_array_equal(_ihist([2, 2], 3, (0, 3)), [0, 0.5, 0.5])

    # Multiple segments, same value
    # 2 samples per pixel -> 5 samples per hist
    assert_allclose(_ihist(0.5*np.ones(5), 3, (0, 1)), [0, 1/4+3/4+3/4+1/4, 0])

    # 3 samples per pixel -> 6 samples per hist
    assert_allclose(_ihist(0.5*np.ones(6), 3, (0, 1)),
                    [0, 1/3+2/3+1+2/3+1/3, 0])

    # 14 samples per pixel -> 29 samples per hist
    assert_allclose(_ihist(0.5*np.ones(29), 3, (0, 1)), [0, 14, 0])
    assert_allclose(_ihist(0.5*np.ones(29), 2, (0, 1)), [7, 7])

    # 15 samples per pixel -> 30 samples per hist
    assert_allclose(_ihist(0.5*np.ones(30), 3, (0, 1)), [0, 15, 0])
    assert_allclose(_ihist(0.5*np.ones(30), 2, (0, 1)), [7.5, 7.5])

    # Multiple segments, linear
    # 2 samples per pixel
    assert_allclose(_ihist([5, 4, 3, 2, 1], 4, (1, 5)),
                    [1/4, 3/4, 3/4, 1/4])

    # TODO: WRITE MORE

    # Random data sums correctly
    np.random.seed(42)  # deterministic tests
    S = 100 if SLOW_TESTS else 1
    for N in np.random.exponential(100000, size=S).astype('int'):
        if SLOW_TESTS:
            print(N, end=', ')
        bins = np.random.random_integers(1, 3000, 1)[0]  # numpy int32
        chunksize = 2*N if N % 2 else 2*N + 1
        a = np.random.randn(chunksize)
        lower = np.amin(a) - 15*np.random.rand(1)[0]
        upper = np.amax(a) + 14*np.random.rand(1)[0]
        assert_allclose(N, sum(_ihist(a, bins, (lower, upper))))


if __name__ == '__main__':
    from numpy import sin

    t = linspace(0, 20, 48000)
    sig = sin(t**3)*sin(t)

    plt.figure()
    plt.margins(0)
    plt.ylim(-2, 2)
    plt.plot(t, sig)

    scopeplot(sig, range_=2, plot=True)
