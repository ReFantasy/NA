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

real = ti.f32
ti.init(default_fp=real, arch=ti.cpu)

show_gui = False


n_mg_levels = 4
pre_and_post_smoothing = 4
bottom_smoothing = 200

N = 128 * 16
field_offset = 5

N_gui = 512  # gui resolution

pixels = ti.field(dtype=real, shape=(N_gui, N_gui))  # image buffer
h = 2.0 / N  # grid spacing

square_h_inv = 1.0 / (h * h)  # precompute for efficiency

x = ti.field(dtype=real, shape=(N + 2 * field_offset, N + 2 * field_offset), offset=(-field_offset, -field_offset))
p = ti.field(dtype=real, shape=(N + 2 * field_offset, N + 2 * field_offset), offset=(-field_offset, -field_offset))
Ap = ti.field(dtype=real, shape=(N + 2 * field_offset, N + 2 * field_offset), offset=(-field_offset, -field_offset))


alpha = ti.field(dtype=real, shape=())  # step size
beta = ti.field(dtype=real, shape=())  # step size
sum_ = ti.field(dtype=real, shape=())  # storage for reductions


r, z, z_temp = [], [], []
for lvl in range(n_mg_levels):
    size = N // (2**lvl) + 2 * field_offset
    r_lvl = ti.field(dtype=real, shape=(size, size), offset=(-field_offset, -field_offset))
    z_lvl = ti.field(dtype=real, shape=(size, size), offset=(-field_offset, -field_offset))
    z_temp_lvl = ti.field(dtype=real, shape=(size, size), offset=(-field_offset, -field_offset))
    r.append(r_lvl)
    z.append(z_lvl)
    z_temp.append(z_temp_lvl)


# @ti.func
# def idx(tensor, N, i, j):
#     v = 0.0
#     if i < 0 or i >= N or j < 0 or j >= N:
#         v = 0.0
#     else:
#         v = tensor[i, j]
#     return v


@ti.kernel
def reduce(p_: ti.template(), q_: ti.template()):
    # for I in ti.grouped(p_):
    #     sum_[None] += p_[I] * q_[I]
    for i, j in ti.ndrange(N, N):
        sum_[None] += p_[i, j] * q_[i, j]


@ti.kernel
def compute_Ap():
    for i, j in ti.ndrange(N, N):
        # A is implicitly expressed as a 3-D laplace operator
        Ap[i, j] = (4.0 * p[i, j] - p[i + 1, j] - p[i - 1, j] - p[i, j + 1] - p[i, j - 1]) * square_h_inv


@ti.kernel
def update_x():
    # for I in ti.grouped(p):
    #     x[I] += alpha[None] * p[I]
    for i, j in ti.ndrange(N, N):
        x[i, j] += alpha[None] * p[i, j]


@ti.kernel
def update_r():
    # for I in ti.grouped(p):
    #     r[0][I] -= alpha[None] * Ap[I]
    for i, j in ti.ndrange(N, N):
        r[0][i, j] -= alpha[None] * Ap[i, j]


@ti.kernel
def update_p():
    # for I in ti.grouped(p):
    #     p[I] = z[0][I] + beta[None] * p[I]
    for i, j in ti.ndrange(N, N):
        p[i, j] = z[0][i, j] + beta[None] * p[i, j]


@ti.kernel
def init():
    # for i, j in ti.ndrange((N_ext, N_tot - N_ext), (N_ext, N_tot - N_ext)):
    #     x[i, j] = 0.0
    #     # r = b - Ax, where x = 0; therefore r = b
    #     r[0][i, j] = 1.0  # f(x) = 1.0 for Poisson equation with zero Dirichlet boundary conditions
    #     z[0][i, j] = 0.0  # initial preconditioned residual z0 = r0
    #     p[i, j] = 0.0  # initial search direction p0 = r0
    #     Ap[i, j] = 0.0
    for I in ti.grouped(x):
        x[I] = 0.0
        # r = b - Ax, where x = 0; therefore r = b
        r[0][I] = 1.0  # f(x) = 1.0 for Poisson equation with zero Dirichlet boundary conditions
        z[0][I] = 0.0  # initial preconditioned residual z0 = r0
        p[I] = 0.0  # initial search direction p0 = r0
        Ap[I] = 0.0


@ti.kernel
def rbgs(l: ti.template(), phase: ti.template()):
    # solve A z = r approximately by performing a few iterations of red-black Gauss-Seidel relaxation, where A is the 32-D Laplace operator
    # phase = red/black Gauss-Seidel phase
    # for i, j in r[l]:
    #     if (i + j) & 1 == phase:
    #         z[l][i, j] = (r[l][i, j] * h * h + z[l][i + 1, j] + z[l][i - 1, j] + z[l][i, j + 1] + z[l][i, j - 1]) / 4.0
    # z[l][i, j] = (
    #     r[l][i, j] * h * h
    #     + idx(z[l], r[l].shape[0], i + 1, j)
    #     + idx(z[l], r[l].shape[0], i - 1, j)
    #     + idx(z[l], r[l].shape[0], i, j + 1)
    #     + idx(z[l], r[l].shape[0], i, j - 1)
    # ) / 4.0
    for i, j in ti.ndrange(r[l].shape[0] - 2 * field_offset, r[l].shape[1] - 2 * field_offset):
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
    for i, j in r[l]:
        # 纯 Jacobi 的一步估计值: x_star = (b - (L+U)x_old) / D
        jacobi_val = (r[l][i, j] * h * h + z[l][i + 1, j] + z[l][i - 1, j] + z[l][i, j + 1] + z[l][i, j - 1]) / 4.0
        # 阻尼更新公式: x_new = (1 - omega) * x_old + omega * x_star
        z_temp[l][i, j] = (1.0 - omega) * z[l][i, j] + omega * jacobi_val
    # 第二步：将新结果覆盖回 z[l]
    for i, j in z[l]:
        z[l][i, j] = z_temp[l][i, j]


@ti.kernel
def restrict(l: ti.template()):
    # for i, j in r[l]:
    #     res = (
    #         r[l][i, j]
    #         - (4.0 * z[l][i, j] - z[l][i + 1, j] - z[l][i - 1, j] - z[l][i, j + 1] - z[l][i, j - 1]) * square_h_inv
    #     )
    #     # res = (
    #     #     r[l][i, j]
    #     #     - (
    #     #         4.0 * idx(z[l], r[l].shape[0], i, j)
    #     #         - idx(z[l], r[l].shape[0], i + 1, j)
    #     #         - idx(z[l], r[l].shape[0], i - 1, j)
    #     #         - idx(z[l], r[l].shape[0], i, j + 1)
    #     #         - idx(z[l], r[l].shape[0], i, j - 1)
    #     #     )
    #     #     * square_h_inv
    #     # )

    #     r[l + 1][i // 2, j // 2] += res * 0.25
    for i, j in ti.ndrange(r[l].shape[0] - 2 * field_offset, r[l].shape[1] - 2 * field_offset):
        res = (
            r[l][i, j]
            - (4.0 * z[l][i, j] - z[l][i + 1, j] - z[l][i - 1, j] - z[l][i, j + 1] - z[l][i, j - 1]) * square_h_inv
        )
        r[l + 1][i // 2, j // 2] += res * 0.25


@ti.kernel
def prolongate(l: ti.template()):
    for I in ti.grouped(z[l]):
        z[l][I] += z[l + 1][I // 2] * 4.0
    # for i, j in ti.ndrange(z[l].shape[0]-2*field_offset, z[l].shape[1]-2*field_offset):
    #     z[l][i, j] += z[l + 1][i // 2, j // 2] * 4.0


def apply_preconditioner():
    z[0].fill(0)

    for l in range(n_mg_levels - 1):
        for _ in range(pre_and_post_smoothing << l):
            smooth_rbgs(l, 0)
            # smooth_jacobi(l)
        z[l + 1].fill(0)
        r[l + 1].fill(0)
        restrict(l)

    # solve A z = r approximately on the coarsest level by performing a few iterations of red-black Gauss-Seidel relaxation
    for _ in range(bottom_smoothing):
        smooth_rbgs(n_mg_levels - 1, 0)
        # smooth_jacobi(n_mg_levels - 1)

    for l in reversed(range(n_mg_levels - 1)):
        prolongate(l)
        for i in range(pre_and_post_smoothing << l):
            smooth_rbgs(l, 1)
            # smooth_jacobi(l)


@ti.kernel
def paint():
    for i, j in pixels:
        ii = int(i * N / N_gui)
        jj = int(j * N / N_gui)
        pixels[i, j] = x[ii, jj]


if show_gui:
    gui = ti.GUI("Multigrid Preconditioned Conjugate Gradient (MGPCG)", res=(N_gui, N_gui))


def main():

    init()

    sum_[None] = 0.0
    reduce(r[0], r[0])
    rTr_initial = sum_[None]

    apply_preconditioner()

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

        # 将原始 2D 数组直接传给 imshow，并指定 colormap
        im = plt.imshow(pixels_np, cmap="viridis")
        # 添加颜色条来显示真实的数据范围
        plt.colorbar(im, label="solution")

        plt.axis("off")

        # 自动调整布局，使图像和颜色条更紧凑、防止标签被截断
        plt.tight_layout()

        # 保存为高清图片，dpi 即为 PPI（每英寸像素点数），通常 300 或 600 用于高清/出版
        # plt.savefig("mgpcg_poisson_2d.png", dpi=600)
        plt.show()
