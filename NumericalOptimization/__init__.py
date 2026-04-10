import jax

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


from . import linear_search
from . import gradient_methods
from . import utils
