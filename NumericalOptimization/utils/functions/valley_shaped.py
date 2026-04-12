# -------------------------------------------------------------------------
#
# SIX-HUMP CAMEL FUNCTION
#
# Authors: Sonja Surjanovic, Simon Fraser University
#          Derek Bingham, Simon Fraser University
# Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
#
# Copyright 2013. Derek Bingham, Simon Fraser University.
#
# THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
# FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
# derivative works, such modified software should be clearly marked.
# Additionally, this program is free software; you can redistribute it
# and/or modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation; version 2.0 of the License.
# Accordingly, this program is distributed in the hope that it will be
# useful, but WITHOUT ANY WARRANTY; without even the implied warranty
# of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# For function details and reference information, see:
# http://www.sfu.ca/~ssurjano/
#
# -------------------------------------------------------------------------
#
# INPUTS:
#
# xx = [x1, x2]
#
# -------------------------------------------------------------------------

import jax
import jax.numpy as jnp


@jax.jit
def camel6(xx):
    x1, x2 = xx
    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2

    y = term1 + term2 + term3
    return y


if __name__ == "__main__":
    import NumericalOptimization as optimizer
    from NumericalOptimization.utils import LineSearchFunction
    from NumericalOptimization.utils.draw import draw2d

    xstar, fstar, k = optimizer.gradient_methods.quasi_newton(
        camel6,
        jnp.array([-2.5, 2.0]),
        epsilon=1e-6,
        line_search_function=LineSearchFunction(),
    )
    print(f"xstar: {xstar}, fstar: {fstar}, k: {k}")

    draw2d(camel6, x_range=(-2, 2), y_range=(-1, 1), samples_x=120, samples_y=60)
