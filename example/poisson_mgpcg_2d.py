"""
Multigrid Preconditioned Conjugate Gradient (CG) method for solving the 2D Poisson equation on the square domain (-1,1)x(-1,1) with zero Dirichlet boundary conditions.
The Poisson equation is given by:
    -Δu = f
where Δ is the Laplace operator, u is the unknown function we want to solve for, and f is the source term. In this example, we set f(x) = 1.
The example is from the dealii documentation: https://dealii.org/current/doxygen/deal.II/step_3.html
"""

import taichi as ti
import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser(
    description="Multigrid Preconditioned Conjugate Gradient (CG) method for solving the 2D Poisson equation."
)
parser.add_argument("-s", "--show-gui", action="store_true", help="Show the GUI with the solution.")
parser.add_argument("-a", "--arch", type=str, default="CPU", help="Taichi architecture to use (e.g., 'cpu', 'gpu', 'cuda', 'vulkan', 'metal').")
parser.add_argument("-l", "--mg-levels", type=int, default=8, help="Number of multigrid levels.")
parser.add_argument("--pre-post-smoothing", type=int, default=3, help="Number of smoothing iterations for pre- and post-smoothing at each level (will be multiplied by 2^l for level l).")
parser.add_argument("--bottom-smoothing", type=int, default=50, help="Number of smoothing iterations for the coarsest level solve.")
parser.add_argument("-m", "--smoothing-method", type=str, default="rbgs", choices=["rbgs", "jacobi"], help="Smoothing method to use: 'rbgs' for red-black Gauss-Seidel, 'jacobi' for dampened Jacobi.")
parser.add_argument("-N", "--N", type=int, default=128 * 8, help="Grid resolution (N x N).")
parser.add_argument('-f', "--float", type=str, default="32", choices=["32", "64"], help="Floating point precision to use: 'f32' for 32-bit, 'f64' for 64-bit.")
args = parser.parse_args()

if args.float == "32":
    real = ti.f32
elif args.float == "64":
    real = ti.f64

if args.arch.lower() == "cpu":
    ti.init(default_fp=real, arch=ti.cpu)
elif args.arch.lower() == "gpu":
    ti.init(default_fp=real, arch=ti.gpu)
elif args.arch.lower() == "cuda":
    ti.init(default_fp=real, arch=ti.cuda)
elif args.arch.lower() == "vulkan":
    ti.init(default_fp=real, arch=ti.vulkan)
elif args.arch.lower() == "metal":
    ti.init(default_fp=real, arch=ti.metal)
else:
    raise ValueError(
        f"Unsupported architecture: {args.arch}. Supported architectures are: 'cpu', 'gpu', 'cuda', 'vulkan', 'metal'."
    )

# --------------------------------------------------------------------------------------------------------
# Multigrid and CG solver parameters, including the number of multigrid levels,
# the number of smoothing iterations for pre- and post-smoothing,
# the number of smoothing iterations for the coarsest level solve,
# the grid resolution (N x N), and the padding size for the fields to handle boundary conditions,
# as well as the grid spacing (h) and its inverse squared for efficient computation of the Laplace operator
# --------------------------------------------------------------------------------------------------------
mg_levels = args.mg_levels
pre_and_post_smoothing = args.pre_post_smoothing
bottom_smoothing = args.bottom_smoothing
N = args.N
padding = 5  # padding size for fields to handle boundary conditions; we need to access neighbors up to 2 grid points away for the 2-D Laplace operator, therefore a padding of 5 is sufficient to avoid out-of-bounds access during the matrix-free operator application and smoothing steps
h = 2.0 / (N - 1)  # grid spacing
square_h_inv = 1.0 / (h * h)  # precompute for efficiency

# --------------------------------------------------------------------------------------------------------
# Fields for the solution (x), search direction (p), matrix-vector product (Ap), residuals (r),
# preconditioned residuals (z), and temporary storage for smoothing (z_temp)
# --------------------------------------------------------------------------------------------------------
x = ti.field(dtype=real, shape=(N + 2 * padding, N + 2 * padding), offset=(-padding, -padding))
p = ti.field(dtype=real, shape=(N + 2 * padding, N + 2 * padding), offset=(-padding, -padding))
Ap = ti.field(dtype=real, shape=(N + 2 * padding, N + 2 * padding), offset=(-padding, -padding))

r, z, z_temp = [], [], []
for lvl in range(mg_levels):
    size = N // (2**lvl) + 2 * padding
    r_lvl = ti.field(dtype=real, shape=(size, size), offset=(-padding, -padding))
    z_lvl = ti.field(dtype=real, shape=(size, size), offset=(-padding, -padding))
    z_temp_lvl = ti.field(dtype=real, shape=(size, size), offset=(-padding, -padding))
    r.append(r_lvl)
    z.append(z_lvl)
    z_temp.append(z_temp_lvl)

alpha = ti.field(dtype=real, shape=())  # step size
beta = ti.field(dtype=real, shape=())  # step size
sum_ = ti.field(dtype=real, shape=())  # storage for reductions


# --------------------------------------------------------------------------------------------------------
# Matrix-free operator application kernel for computing Ap = A p, where A is the 2-D Laplace operator
# --------------------------------------------------------------------------------------------------------
@ti.kernel
def compute_Ap():
    for i, j in ti.ndrange(N, N):
        # A is implicitly expressed as a 2-D laplace operator
        Ap[i, j] = (4.0 * p[i, j] - p[i + 1, j] - p[i - 1, j] - p[i, j + 1] - p[i, j - 1]) * square_h_inv


# --------------------------------------------------------------------------------------------------------
# auxiliary kernels for vector operations in the CG solver loop,
# such as dot product reduction and vector updates
# --------------------------------------------------------------------------------------------------------
@ti.kernel
def reduce(p_: ti.template(), q_: ti.template()):
    for i, j in ti.ndrange(N, N):
        sum_[None] += p_[i, j] * q_[i, j]


@ti.kernel
def update_x():
    for i, j in ti.ndrange(N, N):
        x[i, j] += alpha[None] * p[i, j]


@ti.kernel
def update_r():
    for i, j in ti.ndrange(N, N):
        r[0][i, j] -= alpha[None] * Ap[i, j]


@ti.kernel
def update_p():
    for i, j in ti.ndrange(N, N):
        p[i, j] = z[0][i, j] + beta[None] * p[i, j]


# --------------------------------------------------------------------------------------------------------
# Multigrid V-cycle components: smoothing (red-black Gauss-Seidel or dampened Jacobi)
# --------------------------------------------------------------------------------------------------------
@ti.kernel
def rbgs(l: ti.template(), phase: ti.template()):
    # solve A z = r approximately by performing a few iterations of red-black Gauss-Seidel relaxation, where A is the 2-D Laplace operator
    # phase = red/black Gauss-Seidel phase
    for i, j in ti.ndrange(r[l].shape[0] - 2 * padding, r[l].shape[1] - 2 * padding):
        if (i + j) & 1 == phase:
            z[l][i, j] = (r[l][i, j] * h * h + z[l][i + 1, j] + z[l][i - 1, j] + z[l][i, j + 1] + z[l][i, j - 1]) / 4.0


def smooth_rbgs(l: ti.template(), dir: int):
    if dir == 0:
        rbgs(l, 0)  # red-black Gauss-Seidel phase 0
        rbgs(l, 1)  # red-black Gauss-Seidel phase 1
    else:
        rbgs(l, 1)  # red-black Gauss-Seidel phase 1
        rbgs(l, 0)  # red-black Gauss-Seidel phase 0


@ti.kernel
def smooth_jacobi(l: ti.template()):
    # dampened Jacobi relaxation, which is more parallelizable than Gauss-Seidel
    # 第一步：计算 Jacobi 更新并存入临时场
    omega = 0.667  # 阻尼因子，通常在 (0,1) 之间
    for i, j in ti.ndrange(r[l].shape[0] - 2 * padding, r[l].shape[1] - 2 * padding):
        # 纯 Jacobi 的一步估计值: x_star = (b - (L+U)x_old) / D
        jacobi_val = (r[l][i, j] * h * h + z[l][i + 1, j] + z[l][i - 1, j] + z[l][i, j + 1] + z[l][i, j - 1]) / 4.0
        # 阻尼更新公式: x_new = (1 - omega) * x_old + omega * x_star
        z_temp[l][i, j] = (1.0 - omega) * z[l][i, j] + omega * jacobi_val
    # 第二步：将新结果覆盖回 z[l]
    for i, j in ti.ndrange(z[l].shape[0] - 2 * padding, z[l].shape[1] - 2 * padding):
        z[l][i, j] = z_temp[l][i, j]

def smooth(l: ti.template(), dir: int, method: str = "rbgs"):
    # you can choose either red-black Gauss-Seidel or dampened Jacobi for smoothing; 
    # red-black Gauss-Seidel typically converges faster but is less parallelizable than Jacobi
    if method == "rbgs":
        smooth_rbgs(l, dir)
    elif method == "jacobi":
        smooth_jacobi(l)

# --------------------------------------------------------------------------------------------------------
# Multigrid V-cycle components: restriction and prolongation
# --------------------------------------------------------------------------------------------------------
@ti.kernel
def restrict(l: ti.template()):
    for i, j in ti.ndrange(r[l].shape[0] - 2 * padding, r[l].shape[1] - 2 * padding):
        res = (
            r[l][i, j]
            - (4.0 * z[l][i, j] - z[l][i + 1, j] - z[l][i - 1, j] - z[l][i, j + 1] - z[l][i, j - 1]) * square_h_inv
        )
        r[l + 1][i // 2, j // 2] += res * 0.25


@ti.kernel
def prolongate(l: ti.template()):
    for I in ti.grouped(z[l]):
        z[l][I] += z[l + 1][I // 2] * 4.0


# --------------------------------------------------------------------------------------------------------
# Multigrid V-cycle component: applying the multigrid preconditioner,
# which consists of recursively performing V-cycles to approximately solve A z = r,
# where A is the 2-D Laplace operator
# --------------------------------------------------------------------------------------------------------
def apply_preconditioner():
    z[0].fill(0)

    for l in range(mg_levels - 1):
        for _ in range(pre_and_post_smoothing << l):
            smooth(l, 0, method=args.smoothing_method)
        z[l + 1].fill(0)
        r[l + 1].fill(0)
        restrict(l)

    # solve A z = r approximately on the coarsest level by performing a few iterations of red-black Gauss-Seidel relaxation
    for _ in range(bottom_smoothing):
        smooth(mg_levels - 1, 0, method=args.smoothing_method)

    for l in reversed(range(mg_levels - 1)):
        prolongate(l)
        for i in range(pre_and_post_smoothing << l):
            smooth(l, 1, method=args.smoothing_method)


# --------------------------------------------------------------------------------------------------------
# Initialization and GUI setup
# --------------------------------------------------------------------------------------------------------
@ti.kernel
def init():
    x.fill(0)
    r[0].fill(0)
    p.fill(0)
    for i, j in ti.ndrange(N, N):
        # r = b - Ax, where x = 0; therefore r = b
        r[0][i, j] = 1.0  # f(x) = 1.0 for Poisson equation with zero Dirichlet boundary conditions

show_gui = args.show_gui
gui_resolution = 512  # gui resolution
pixels = ti.field(dtype=real, shape=(gui_resolution, gui_resolution))  # image buffer


@ti.kernel
def paint():
    for i, j in pixels:
        ii = int(i * N / gui_resolution)
        jj = int(j * N / gui_resolution)
        pixels[i, j] = x[ii, jj]


if show_gui:
    gui = ti.GUI("Multigrid Preconditioned Conjugate Gradient (MGPCG)", res=(gui_resolution, gui_resolution))


# --------------------------------------------------------------------------------------------------------
# Main MGPCG solver loop
# --------------------------------------------------------------------------------------------------------
def main():

    init()

    sum_[None] = 0.0
    reduce(r[0], r[0])
    rTr_initial = sum_[None]

    apply_preconditioner()

    # update_p: p = z + beta * p, where p is initialized to zero and beta is not used in the first iteration,
    # therefore p = z here in the first iteration; in subsequent iterations, p is updated using the computed beta value from the previous iteration's rTz and rTz_old values
    update_p()

    sum_[None] = 0.0
    reduce(r[0], z[0])
    rTz_old = sum_[None]

    k = 0
    t1 = time.time()
    while True:
        compute_Ap()

        sum_[None] = 0.0
        reduce(p, Ap)
        pAp = sum_[None]
        alpha[None] = rTz_old / pAp

        # x = x + alpha p
        update_x()

        # r = r - alpha Ap
        update_r()

        sum_[None] = 0.0
        reduce(r[0], r[0])
        rTr = sum_[None]
        print(f"iter {k}: residual: {rTr:.6e}")
        if rTr < 2e-8:  # rTr_initial * 1e-12:
            print(f"Converged! Final residual: {rTr:.6e}, initial residual: {rTr_initial:.6e}")
            break

        # z = M^{-1} r, where M is the multigrid preconditioner
        apply_preconditioner()

        sum_[None] = 0.0
        reduce(r[0], z[0])
        rTz = sum_[None]

        beta[None] = rTz / rTz_old

        # p = z + beta p
        update_p()

        rTz_old = rTz
        k += 1

        if show_gui:
            paint()
            gui.set_image(pixels)
            gui.show()

    t2 = time.time()
    print(f"Total time: {t2 - t1:.4f} seconds")


if __name__ == "__main__":

    main()

    if show_gui:
        pixels_np = pixels.to_numpy()
        plt.title("Multigrid Preconditioned Conjugate Gradient (MGPCG)")
        plt.gcf().canvas.manager.set_window_title("Multigrid Preconditioned Conjugate Gradient (MGPCG)")
        plt.rcParams["font.family"] = "Times New Roman"
        im = plt.imshow(pixels_np, cmap="viridis")
        plt.colorbar(im, label="solution")
        plt.axis("off")
        plt.tight_layout()
        # plt.savefig("mgpcg_poisson_2d.png", dpi=600)
        plt.show()
