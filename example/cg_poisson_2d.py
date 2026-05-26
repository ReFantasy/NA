"""
Conjugate Gradient (CG) method for solving the 2D Poisson equation on the square domain (-1,1)x(-1,1) with zero Dirichlet boundary conditions.
The Poisson equation is given by:
    -Δu = f
where Δ is the Laplace operator, u is the unknown function we want to solve for, and f is the source term. In this example, we set f(x) = 1.
The example is from the dealii documentation: https://dealii.org/current/doxygen/deal.II/step_3.html
"""

import taichi as ti
import matplotlib.pyplot as plt

real = ti.f32
ti.init(default_fp=real, arch=ti.x64, kernel_profiler=False)

# grid parameters
N_ext = 4  # number of ghost cells for boundary conditions
N = 128 * N_ext
N_TOL = N + N_ext * 2

N_gui = 512  # gui resolution
pixels = ti.field(dtype=real, shape=(N_gui, N_gui))  # image buffer
h = 2.0 / (N - 1)  # grid spacing
square_h_inv = 1.0 / (h * h)  # precompute for efficiency

# setup sparse simulation data arrays
x = ti.field(dtype=real)  # solution
p = ti.field(dtype=real)  # conjugate gradient
Ap = ti.field(dtype=real)  # matrix-vector product
r = ti.field(dtype=real)  # residual
alpha = ti.field(dtype=real)  # step size
beta = ti.field(dtype=real)  # step size
sum_ = ti.field(dtype=real)  # storage for reductions

ti.root.pointer(ti.ij, [N // N_ext + 2]).dense(ti.ij, N_ext).place(x, p, Ap, r)
ti.root.place(alpha, beta, sum_)


@ti.kernel
def reduce(p_: ti.template(), q_: ti.template()):
    for I in ti.grouped(p_):
        sum_[None] += p_[I] * q_[I]


@ti.kernel
def compute_Ap():
    for i, j in Ap:
        # A is implicitly expressed as a 3-D laplace operator
        Ap[i, j] = (4.0 * p[i, j] - p[i + 1, j] - p[i - 1, j] - p[i, j + 1] - p[i, j - 1]) * square_h_inv


@ti.kernel
def update_x():
    for I in ti.grouped(p):
        x[I] += alpha[None] * p[I]


@ti.kernel
def update_r():
    for I in ti.grouped(p):
        r[I] -= alpha[None] * Ap[I]


@ti.kernel
def update_p():
    for I in ti.grouped(p):
        p[I] = r[I] + beta[None] * p[I]


@ti.kernel
def init():
    for i, j in ti.ndrange((N_ext, N_ext + N), (N_ext, N_ext + N)):
        x[i, j] = 0.0
        # r = b - Ax, where x = 0; therefore r = b
        r[i, j] = 1.0  # f(x) = 1.0 for Poisson equation with zero Dirichlet boundary conditions
        p[i, j] = r[i, j]  # initial search direction p0 = r0


@ti.kernel
def paint():
    for i, j in pixels:
        ii = int(i * N / N_gui)
        jj = int(j * N / N_gui)
        pixels[i, j] = x[ii + N_ext, jj + N_ext]  # * 2.0


gui = ti.GUI("Conjugate Gradient (CG) for 2D Poisson equation", res=(N_gui, N_gui))


def main():

    init()

    reduce(r, r)
    rTr_initial = sum_[None]
    rTr_old = sum_[None]

    k = 0
    while gui.running:
        compute_Ap()

        sum_[None] = 0.0
        reduce(p, Ap)
        pAp = sum_[None]
        alpha[None] = rTr_old / pAp

        # x = x + alpha p
        update_x()

        # r = r - alpha Ap
        update_r()

        sum_[None] = 0.0
        reduce(r, r)
        rTr = sum_[None]

        print(f"iter {k}: residual: {rTr:.6e}")
        if rTr < 2e-8:  # rTr_initial * 1e-12:
            print(f"Converged! Final residual: {rTr:.6e}, initial residual: {rTr_initial:.6e}")
            break

        beta[None] = rTr / rTr_old

        # p = z + beta p
        update_p()

        rTr_old = rTr
        k += 1

        paint()
        gui.set_image(pixels)
        gui.show()
    pass


if __name__ == "__main__":
    main()

    pixels_np = pixels.to_numpy()

    plt.title("Conjugate Gradient (CG) for 2D Poisson equation")
    plt.gcf().canvas.manager.set_window_title("Conjugate Gradient (CG) for 2D Poisson equation")
    plt.rcParams["font.family"] = "Times New Roman"

    # 将原始 2D 数组直接传给 imshow，并指定 colormap
    im = plt.imshow(pixels_np, cmap="viridis")
    # 添加颜色条来显示真实的数据范围
    plt.colorbar(im, label="solution")

    plt.axis("off")

    # 自动调整布局，使图像和颜色条更紧凑、防止标签被截断
    plt.tight_layout()

    # 保存为高清图片，dpi 即为 PPI（每英寸像素点数），通常 300 或 600 用于高清/出版
    # plt.savefig("cg_poisson_2d.png", dpi=600)
    plt.show()
