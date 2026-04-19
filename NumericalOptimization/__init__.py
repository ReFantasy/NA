import os
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true"
import jax

jax.config.update("jax_platforms", "cpu")
jax.config.update("jax_enable_x64", True)


from . import linear_search
from . import gradient_methods
from . import utils
from . import constrained_optimization
