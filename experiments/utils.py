import gpytoolbox
import numpy as np
from meshplot import plot as plot_mesh
import matplotlib.pyplot as plt
  


def normalize_points(v):
    """
    Normalize points to tightly fit in [-0.5, 0.5]^d

    v: (n, d)

    returns: (n, d)
    """
    v = v - v.min(0)
    v = v / v.max()
    v = v - 0.5 * v.max(0)
    return v


def face_normals(v, f):
    v1 = v[f[:, 1]] - v[f[:, 0]]
    v2 = v[f[:, 2]] - v[f[:, 1]]
    n = np.cross(v1, v2) / 2
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)

    return n


def face_areas(v, f):
    v1 = v[f[:, 1]] - v[f[:, 0]]
    v2 = v[f[:, 2]] - v[f[:, 1]]
    n = np.cross(v1, v2) / 2

    a = np.linalg.norm(n, axis=-1)
    return a


def points_from_barycentric(v, f, pfi, pb):
    p = (v[f[pfi]] * pb[..., None]).sum(1)
    fn = face_normals(v, f)
    n = fn[pfi]

    return p, n


def uniform_sample_mesh(v, f, n, rng):
    a = face_areas(v, f)
    pfi = rng.choice(f.shape[0], size=n, p=a / a.sum())
    pb = rng.dirichlet(np.ones(3), size=n)

    return points_from_barycentric(v, f, pfi, pb)


def rotation_between_vectors(v1, v2):
    """
    https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    """
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    if any(v):
        cos = np.dot(v1, v2)
        sin = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - cos) / (sin**2))

    return np.eye(3)


def simulate_scan(v, f, cam_origins, cam_directions, angle, density):
    """
    Expects rays to be packed as a tensor of rows of [origin, direction]
    """
    extent = np.tan(angle)
    rays = np.mgrid[(slice(-extent, extent, density * 1j),) * 2].reshape(2, -1).T
    rays = np.concatenate([rays, np.ones((rays.shape[0], 1))], axis=-1)
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True)

    ray_origins = cam_origins.repeat(rays.shape[0], axis=0)
    e3 = np.array([0.0, 0.0, 1.0])

    rotations = np.stack(
        [
            rotation_between_vectors(e3, cam_directions[i])
            for i in range(cam_directions.shape[0])
        ],
        axis=0,
    )
    ray_directions = (rays @ rotations.transpose(0, 2, 1)).reshape(-1, 3)

    t, pfi, pb = gpytoolbox.ray_mesh_intersect(
        ray_origins,
        ray_directions,
        v,
        f,
    )

    hit = t != np.inf

    return points_from_barycentric(v, f, pfi[hit], pb[hit])


def grid_2d_mesh(density):
    """
    Mesh corresponding to [-0.5, 0.5]^2 in R^3
    """

    grid = np.mgrid[(slice(-0.5, 0.5, density * 1j),) * 2]
    x1 = grid[0]
    x2 = grid[1]

    v = np.stack([x1, x2, np.zeros_like(x1)], axis=-1).reshape(-1, 3)

    fi = np.mgrid[0 : density - 1, 0 : density - 1].reshape(2, -1).T

    v1 = fi[:, 0] * 100 + fi[:, 1]
    v2 = v1 + 1
    v3 = (fi[:, 0] + 1) * 100 + fi[:, 1]
    v4 = v3 + 1

    f1 = np.stack([v1, v2, v4], axis=-1)
    f2 = np.stack([v1, v4, v3], axis=-1)
    f = np.concatenate([f1, f2], axis=0)

    return v, f, (x1, x2)



def plot_mean(name, x1, x2, f, save=True):
    plt.figure(figsize=(4, 4))
    plt.pcolormesh(x1, x2, f.reshape(x1.shape), shading="Gouraud", cmap="RdBu")
    plt.contour(x1, x2, f.reshape(x1.shape), np.array([0.0]))
    plt.axis("equal")
    plt.axis("off")
    if save:
        plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=400)
    else:
        plt.show()


def plot_variance(name, x1, x2, f, save=True):
    plt.figure(figsize=(4, 4))
    plt.pcolormesh(x1, x2, f.reshape(x1.shape), shading="Gouraud", cmap="plasma")
    plt.axis("equal")
    plt.axis("off")
    if save:
        plt.savefig(name, bbox_inches="tight", pad_inches=0, dpi=400)
    else:
        plt.show()


def plot_mesh_cloud_grid(mesh, cloud=None, grid=None):
    v, f = mesh
    plot = plot_mesh(v, f)

    if cloud:
        if len(cloud) == 2:
            p, n = cloud
            plot.add_points(p, shading={"point_size": 0.1})
            plot.add_points(
                p + n * 0.025, shading={"point_size": 0.1, "point_color": "green"}
            )
            plot.add_points(
                p + n * 0.05, shading={"point_size": 0.1, "point_color": "blue"}
            )
        else:
            plot.add_points(cloud, shading={"point_size": 0.1})

    if grid:
        v_grid, f_grid = grid
        plot.add_mesh(v_grid, f_grid)

    return plot
