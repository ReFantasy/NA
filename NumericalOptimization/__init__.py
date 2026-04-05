import jax

#jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)

from .utils import *
from . import linear_search
