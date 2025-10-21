import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -------------------------------
# CONFIGURATION
# -------------------------------
num_points = 6          # number of Voronoi sites
grid_size = 300         # resolution (higher = smoother, slower)
frames = 100            # number of frames per animation loop
growth_rate = 0.15      # how fast the regions grow per frame

# Create grid
x = np.linspace(0, 10, grid_size)
y = np.linspace(0, 10, grid_size)
X, Y = np.meshgrid(x, y)

# Initialize globals for distance maps (we’ll fill these later)
dist_euclid = np.zeros((num_points, grid_size, grid_size))
dist_manhat = np.zeros((num_points, grid_size, grid_size))

# Keep a reference to scatter artists so we can remove them if desired
scatter_artists = []

# -------------------------------
# FUNCTION: Generate new random points & recompute distances
# -------------------------------
def reset_points():
    """Generate new random points and recompute distances for both metrics."""
    global points, dist_euclid, dist_manhat, region_colors, scatter_artists

    points = np.random.rand(num_points, 2) * 10  # new random sites

    # Precompute distance grids
    for i, (px, py) in enumerate(points):
        dist_euclid[i] = np.sqrt((X - px)**2 + (Y - py)**2)
        dist_manhat[i] = np.abs(X - px) + np.abs(Y - py)

    # Assign a unique set of colors to each site (use modern API to avoid deprecation)
    colors = plt.get_cmap('Pastel2', num_points)
    region_colors = colors(np.arange(num_points))

    # Remove previous scatter artists (if any) and draw the new ones
    # We remove artists by calling .remove() on them
    for art in list(scatter_artists):
        try:
            art.remove()
        except Exception:
            pass
    scatter_artists = []

    for ax in axes:
        sc = ax.scatter(points[:, 0], points[:, 1],
                        c='red', s=50, edgecolor='black', zorder=3)
        scatter_artists.append(sc)


# -------------------------------
# SETUP FIGURE
# -------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
titles = ["Euclidean Growth (circles)", "Manhattan Growth (diamonds)"]

for ax, title in zip(axes, titles):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

# Initial empty images for both diagrams (RGBA)
img_euclid = axes[0].imshow(np.zeros((grid_size, grid_size, 4)),
                            extent=(0, 10, 0, 10), origin='lower')
img_manhat = axes[1].imshow(np.zeros((grid_size, grid_size, 4)),
                            extent=(0, 10, 0, 10), origin='lower')

# -------------------------------
# ANIMATION UPDATE FUNCTION
# -------------------------------
def update(frame):
    # When the animation restarts, randomize the sites
    if frame == 0:
        reset_points()

    # Current "radius" of growth
    r = frame * growth_rate

    # Which site is closest for each pixel
    nearest_euclid = np.argmin(dist_euclid, axis=0)
    nearest_manhat = np.argmin(dist_manhat, axis=0)

    # Pixels that have been reached (within radius r)
    mask_euclid = np.min(dist_euclid, axis=0) <= r
    mask_manhat = np.min(dist_manhat, axis=0) <= r

    # Create colored RGBA arrays (copy so we don't modify original colormap array)
    color_map_e = region_colors[nearest_euclid].copy()
    color_map_m = region_colors[nearest_manhat].copy()

    # Hide unreached pixels (transparent background)
    color_map_e[~mask_euclid] = [0, 0, 0, 0]
    color_map_m[~mask_manhat] = [0, 0, 0, 0]

    img_euclid.set_data(color_map_e)
    img_manhat.set_data(color_map_m)

    return [img_euclid, img_manhat]

# -------------------------------
# CREATE & RUN ANIMATION
# -------------------------------
reset_points()  # initialize with first random set
anim = FuncAnimation(fig, update, frames=frames, interval=100,
                     blit=False, repeat=True)

plt.tight_layout()
plt.show()
