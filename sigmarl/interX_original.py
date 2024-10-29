# Copyright (c) 2024, Chair of Embedded Software (Informatik 11) - RWTH Aachen University.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np


def interX(L1, L2, is_return_points=False):
    """
    Calculate the intersections of two curves.
    Each curve should be a two-row variable, with the first row representing x-coordinate and the second row representing y-coordinate.
    """

    if L1.size == 0 or L2.size == 0:
        return np.array([]).reshape(2, 0) if is_return_points else False

    # Preliminary stuff
    x1, y1 = L1
    x2, y2 = L2
    dx1, dy1 = np.diff(x1), np.diff(y1)
    dx2, dy2 = np.diff(x2), np.diff(y2)

    # Determine 'signed distances'
    S1 = dx1 * y1[:-1] - dy1 * x1[:-1]
    S2 = dx2 * y2[:-1] - dy2 * x2[:-1]

    C1 = D(dx1[:, None] * y2 - dy1[:, None] * x2, S1[:, None]) < 0
    C2 = (D((y1[:, None] * dx2 - x1[:, None] * dy2).T, S2.reshape(-1, 1)) < 0).T

    # Obtain the segments where an intersection is expected
    i, j = np.where(C1 & C2)

    if i.size == 0:
        if is_return_points:
            return np.array([]).reshape(2, 0)
        else:
            return False
    else:
        if is_return_points:
            L = dy2[j] * dx1[i] - dy1[i] * dx2[j]
            nonzero = L != 0
            i, j, L = i[nonzero], j[nonzero], L[nonzero]

            P = np.vstack(
                (
                    (dx2[j] * S1[i] - dx1[i] * S2[j]) / L,
                    (dy2[j] * S1[i] - dy1[i] * S2[j]) / L,
                )
            )
            return np.unique(P, axis=1)
        else:
            return True


def D(x, y):
    return (x[:, :-1] - y) * (x[:, 1:] - y)
