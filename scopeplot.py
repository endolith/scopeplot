# -*- coding: utf-8 -*-
"""
Created on Sun May 24 00:10:45 2015

Density plot of waveforms
"""

from __future__ import division, print_function

import numpy as np
from numpy import asarray, linspace, arange, pad
import matplotlib.pyplot as plt
from scipy.signal import resample

# minimum resampling factor
RS = 2


def _ceildiv(a, b):
    """Ceiling integer division"""
    return -(-a // b)


def _next_regular(target):
    """
    Find the next regular number greater than or equal to target.
    Regular numbers are composites of the prime factors 2, 3, and 5.
    Also known as 5-smooth numbers or Hamming numbers, these are the optimal
    size for inputs to FFTPACK.

    Target must be a positive integer.
    """
    assert abs(int(target)) == target

    if target <= 6:
        return target

    # Quickly check if it's already a power of 2
    if not (target & (target-1)):
        return target

    match = float('inf')  # Anything found will be smaller
    p5 = 1
    while p5 < target:
        p35 = p5
        while p35 < target:
            # Ceiling integer division, avoiding conversion to float
            # (quotient = ceil(target / p35))
            quotient = -(-target // p35)

            # Quickly find next power of 2 >= quotient
            try:
                p2 = 2**((quotient - 1).bit_length())
            except AttributeError:
                # Fallback for Python <2.7
                p2 = 2**(len(bin(quotient - 1)) - 2)

            N = p2 * p35
            if N == target:
                return N
            elif N < match:
                match = N
            p35 *= 3
            if p35 == target:
                return p35
        if p35 < match:
            match = p35
        p5 *= 5
        if p5 == target:
            return p5
    if p5 < match:
        match = p5
    return match


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


def _ihist(a, bins, range):
    """
    interpolated histogram

    a is a sequence of samples, considered to be connected by lines, including
    overlap of neighboring pixels

    bins is number of bins to group them into vertically

    range is total range of output, which may be wider than the widest values
    in a
    """
    a = asarray(a)

    if a.ndim > 1:
        raise AttributeError('a must be a series. it will not be flattened')

    if (not np.isscalar(bins)) or (int(bins) != bins) or bins < 1:
        raise ValueError("`bins` should be a positive integer.")

    if a.size < 2 or a.size % 4 in {0, 3}:
        raise ValueError('not a valid size with overlap: {}'.format(a.size))

    mn, mx = [mi + 0.0 for mi in range]  # Make float

    if a.min() < mn or a.max() > mx:
        raise NotImplementedError("values outside of range are not yet "
                                  "supported {} {}".format(a.min(), a.max()))

    if (mn >= mx):
        raise AttributeError('max must be larger than min in range parameter.')

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

    for n in xrange(len(pairs)):
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
                    raise NotImplementedError('Values on edge of range')
                    # TODO: Could handle this with more ifthens,
                    # but should be a smarter way
            else:
                out[lo-1] += w
        else:
            out[lo-1]    += w * h[n] * (bin_edges[lo] - lower[n])
            out[lo:hi-1] += w * h[n] * bin_width
            out[hi-1]    += w * h[n] * (upper[n] - bin_edges[hi-1])

    return out


def scopeplot(x, width=800, height=400, range=None, cmap=None, plot=None):
    """
    x is the signal to be plotted

    width and height are the pixel dimensions of the output image

    range is the vertical range of the plot.  If a tuple, it is (xmin, xmax),
    if a single number, the range is (-range, range).  If None, it autoscales.

    cmap is colormap.  grayscale by default

    if plot is None, returns the X image array
    if plot is True, plots it directly
    if plot is a string, represents a filename to save the image to
    """

    if cmap is None:
        cmap = 'gray'

    x = asarray(x)

    N = len(x)

    # Add zeros to end to reduce circular Gibbs effects
    MIN_PAD = 5   # TODO: what should this be?  Seems subjective.

    # Make input an optimal length for fast processing
    pad_amount = _next_regular(N + MIN_PAD) - N

    x = pad(x, (0, pad_amount), 'constant')

    # Resample such that signal evenly divides into chunks of equal length
    new_size = int(round(_ceildiv(RS*N, width) * width / N * len(x)))
    print('new size: {}'.format(new_size))

    x = resample(x, new_size)

    if not range:
        range = 1.1 * np.amax(np.abs(x))

    if np.size(range) == 1:
        xmin, xmax = -range, +range
    elif np.size(range) == 2:
        xmin, xmax = range
    else:
        raise ValueError('range not understood')

    spp = _ceildiv(N * RS, width)  # samples per pixel
    norm = 1/spp

    # Pad some zeros at beginning for overlap
    x = pad(x, (spp//2, 0), 'constant')

    X = np.empty((width, height))

    if spp % 2:  # N is odd
        chunksize = 2*spp  # (even)
    else:  # N is even
        chunksize = 2*spp + 1  # (odd)
    print('spp: {}, chunk size: {}'.format(spp, chunksize))

    for n in xrange(0, width):
        chunk = x[n*spp:n*spp+chunksize]
        assert len(chunk)  # don't send empties
        try:
            h = _ihist(chunk, bins=height, range=(xmin, xmax))
        except ValueError:
            print('argh', len(chunk))
        else:
            X[n] = h * norm

    assert np.amax(X) <= 1.001, np.amax(X)

    X = X**(0.4)  # TODO: SUBJECTIVE

    if isinstance(plot, basestring):
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

    assert_allclose(_get_weights(1), [2/2])
    assert_allclose(_get_weights(2), [1/4, 3/4, 3/4, 1/4])
    assert_allclose(_get_weights(3), [2/6, 4/6, 6/6, 4/6, 2/6])
    assert_allclose(_get_weights(4), [1/8, 3/8, 5/8, 7/8, 7/8, 5/8, 3/8, 1/8])
    assert_allclose(_get_weights(5), [2/10, 4/10, 6/10, 8/10, 10/10,
                                      8/10, 6/10, 4/10, 2/10])

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
    if SLOW_TESTS:
        S = 100
    else:
        S = 1
    for N in np.random.exponential(100000, size=S).astype('int'):
        if SLOW_TESTS:
            print(N, end=', ')
        bins = np.random.random_integers(1, 3000, 1)[0]  # numpy int32
        if N % 2:  # N is odd
            chunksize = 2*N  # (even)
        else:  # N is even
            chunksize = 2*N + 1  # (odd)
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

    scopeplot(sig, range=2, plot=True)
