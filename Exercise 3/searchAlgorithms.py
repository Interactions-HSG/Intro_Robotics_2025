import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from queue import PriorityQueue
from heapq import heappush, heappop
import random
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

GRID_SIZE = 20

def neighbors_of(cell, grid):
    """Yield valid neighbor coordinates for a cell in the grid."""
    rows, cols = grid.shape
    x, y = cell
    for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
        nx, ny = x+dx, y+dy
        if 0 <= nx < rows and 0 <= ny < cols and grid[nx, ny] != -1:
            yield (nx, ny)

def create_grid():
    grid = np.zeros((GRID_SIZE, GRID_SIZE))
    for _ in range(3):
        w, h = random.randint(3, 6), random.randint(3, 6)
        x, y = random.randint(0, GRID_SIZE - w - 1), random.randint(0, GRID_SIZE - h - 1)
        grid[x:x+w, y:y+h] = -1
    return grid

def in_bounds(grid, cell):
    rows, cols = grid.shape
    x, y = cell
    return 0 <= x < rows and 0 <= y < cols

def valid_neighbors(cell, grid):
    for nx, ny in neighbors_of(cell, grid):
        yield (nx, ny)

def random_position(grid):
    while True:
        x, y = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
        if grid[x, y] == 0:
            return (x, y)

def dijkstra(grid, start, goal):
    rows, cols = grid.shape
    visited = np.full((rows, cols), False)
    distance = np.full((rows, cols), np.inf)
    previous = np.full((rows, cols, 2), -1)
    
    distance[start] = 0
    pq = PriorityQueue()
    pq.put((0, start))
    
    while not pq.empty():
        dist, current = pq.get()
        if visited[current]:
            continue
        visited[current] = True
        if current == goal:
            break
        for nx, ny in neighbors_of(current, grid):
            if not visited[nx, ny]:
                new_dist = dist + 1
                if new_dist < distance[nx, ny]:
                    distance[nx, ny] = new_dist
                    previous[nx, ny] = [current[0], current[1]]
                    pq.put((new_dist, (nx, ny)))
    
    return _reconstruct_path(previous, start, goal)

def _reconstruct_path(previous, start, goal):
    path = []
    current = goal
    rows, cols = previous.shape[0], previous.shape[1]
    while True:
        path.append(current)
        if current == start:
            break
        x, y = current
        px, py = previous[x, y]
        if px == -1:
            return []
        current = (int(px), int(py))
    path.reverse()
    return path

def dijkstra_steps(grid, start, goal):
    """Run Dijkstra and also record explored/frontier sets for animation.
    Returns: path, steps where steps is list of (explored_set, frontier_set)
    """
    rows, cols = grid.shape
    visited = np.full((rows, cols), False)
    distance = np.full((rows, cols), np.inf)
    previous = np.full((rows, cols, 2), -1)
    distance[start] = 0
    from queue import PriorityQueue
    pq = PriorityQueue()
    pq.put((0, start))

    steps = []
    explored = set()
    frontier = {start}

    while not pq.empty():
        dist, current = pq.get()
        if visited[current]:
            continue
        frontier.discard(current)
        visited[current] = True
        explored.add(current)
        # record step
        steps.append((set(explored), set(frontier)))
        if current == goal:
            break
        for nx, ny in neighbors_of(current, grid):
            if not visited[nx, ny]:
                new_dist = dist + 1
                if new_dist < distance[nx, ny]:
                    distance[nx, ny] = new_dist
                    previous[nx, ny] = [current[0], current[1]]
                    pq.put((new_dist, (nx, ny)))
                    frontier.add((nx, ny))

    path = _reconstruct_path(previous, start, goal)
    return path, steps

def astar_steps(grid, start, goal):
    """A* that returns path and steps (explored/frontier) for animation."""
    heap = [(0, start)]
    visited = {start: None}
    g_score = {start: 0}
    steps = []
    explored = set()
    frontier = {start}

    while heap:
        _, current = heappop(heap)
        frontier.discard(current)
        explored.add(current)
        steps.append((set(explored), set(frontier)))

        if current == goal:
            break

        for nxt in neighbors_of(current, grid):
            if grid[nxt] == 0:
                g = g_score[current] + 1
                if nxt not in g_score or g < g_score[nxt]:
                    g_score[nxt] = g
                    f = g + (abs(nxt[0]-goal[0]) + abs(nxt[1]-goal[1]))
                    heappush(heap, (f, nxt))
                    visited[nxt] = current
                    frontier.add(nxt)

    # reconstruct path
    path = []
    curr = goal
    while curr in visited and curr is not None:
        path.append(curr)
        curr = visited[curr]
    path = path[::-1]
    return path, steps

def run_search(algorithm, grid, start, goal):
    """Selector: algorithm in {'dijkstra','astar'}"""
    if algorithm.lower() == 'dijkstra':
        return dijkstra_steps(grid, start, goal)
    else:
        return astar_steps(grid, start, goal)

class RobotGUI:
    def __init__(self, master):
        self.master = master
        master.title("Mobile Robot Path Planning")
        
        self.grid = create_grid()
        self.start = random_position(self.grid)
        self.goal = random_position(self.grid)
        
        self.fig, self.ax = plt.subplots(figsize=(6,6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack()
        
        self.generate_button = tk.Button(master, text="Generate New Map", command=self.generate_map)
        self.generate_button.pack()
        # Algorithm selection UI (dijkstra or astar)
        self.algorithm_choice = 'dijkstra'
        self.alg_var = tk.StringVar(value=self.algorithm_choice)
        alg_frame = tk.Frame(master)
        alg_frame.pack(pady=4)
        tk.Label(alg_frame, text="Algorithm:").pack(side=tk.LEFT)
        alg_menu = tk.OptionMenu(alg_frame, self.alg_var, 'dijkstra', 'astar', command=self._on_algorithm_change)
        alg_menu.pack(side=tk.LEFT)

        self.robot_marker = None
        self.path_line = None
        self.draw_background()
        self.animate_path()
    
    def generate_map(self):
        self.grid = create_grid()
        self.start = random_position(self.grid)
        self.goal = random_position(self.grid)
        self.draw_background()
        self.animate_path()
    
    def draw_background(self):
        self.ax.clear()
        cmap = colors.ListedColormap(['white', 'black', 'blue', 'green'])
        bounds = [-1, 0, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        # Draw grid + obstacles + start/goal
        display_grid = self.grid.copy()
        display_grid[self.start] = 2
        display_grid[self.goal] = 3
        # keep the image artist so we can update it during animation
        self.img = self.ax.imshow(display_grid, cmap=cmap, norm=norm)
        # Draw graph edges once (use helper to reduce complexity)
        self._draw_grid_edges()
        self.ax.set_title("Robot Navigation with Graph Overlay")

    def _draw_grid_edges(self):
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if self.grid[x, y] == 0:
                    for nx, ny in valid_neighbors((x, y), self.grid):
                        if self.grid[nx, ny] == 0:
                            self.ax.plot([y, ny], [x, nx], color='lightgray', linewidth=0.5)
    
    def animate_path(self):
        algorithm = getattr(self, 'algorithm_choice', 'dijkstra')
        path, steps = run_search(algorithm, self.grid, self.start, self.goal)

        if (not path) and (not steps):
            self.ax.set_title("No path found!")
            self.canvas.draw()
            return

        # base image (start/goal/obstacles)
        base = self.grid.copy()
        base[self.start] = 2
        base[self.goal] = 3

        # temporary artists for exploration visualization
        temp_artists = []

        # animate exploration (frontier + explored)
        for frame_idx, (explored, frontier) in enumerate(steps):
            # update base image (keeps obstacles/start/goal)
            try:
                self.img.set_data(base)
            except Exception:
                self.img = self.ax.imshow(base, origin='upper')

            # remove previous temp artists
            for art in temp_artists:
                try:
                    art.remove()
                except Exception:
                    pass
            temp_artists.clear()

            # plot explored nodes (small cyan squares)
            if explored:
                xs = [c[1] for c in explored]
                ys = [c[0] for c in explored]
                explored_sc = self.ax.scatter(xs, ys, c='#ADD8E6', s=20, marker='s', alpha=0.6)
                temp_artists.append(explored_sc)

            # plot frontier nodes (yellow markers)
            if frontier:
                xs = [c[1] for c in frontier]
                ys = [c[0] for c in frontier]
                frontier_sc = self.ax.scatter(xs, ys, c='yellow', s=30, marker='o', edgecolors='k')
                temp_artists.append(frontier_sc)

            # update title with progress
            explored_count = len(explored)
            self.ax.set_title(f"{algorithm.upper()} — step {frame_idx+1}/{len(steps)} — explored {explored_count}")

            self.canvas.draw()
            self.master.update()
            # pause between frames
            self.master.after(40)

        # after exploration, draw final path and animate robot along it
        for art in temp_artists:
            try:
                art.remove()
            except Exception:
                pass
        temp_artists.clear()

        # draw final path (red line)
        if path:
            px = [p[1] for p in path]
            py = [p[0] for p in path]
            # remove previous path line if present
            if getattr(self, 'path_line', None):
                try:
                    self.path_line.remove()
                except Exception:
                    pass
            self.path_line, = self.ax.plot(px, py, 'r-', linewidth=2, label='Final Path')
            # draw path nodes with a colormap to show progression
            try:
                if getattr(self, 'path_nodes', None):
                    self.path_nodes.remove()
            except Exception:
                pass
            cmap = plt.cm.viridis
            n = len(path)
            if n > 1:
                colors_list = [cmap(i/(n-1)) for i in range(n)]
            else:
                colors_list = [cmap(0.5)]
            xs_nodes = [p[1] for p in path]
            ys_nodes = [p[0] for p in path]
            self.path_nodes = self.ax.scatter(xs_nodes, ys_nodes, c=colors_list, s=60, marker='s', edgecolors='k', zorder=3)
            # title with metrics
            last_explored = len(steps[-1][0]) if steps else 0
            self.ax.set_title(f"{algorithm.upper()} — explored {last_explored} nodes — path length {len(path)}")
            self.canvas.draw()

            # animate robot moving along path
            for pos in path:
                if self.robot_marker:
                    try:
                        self.robot_marker.remove()
                    except Exception:
                        pass
                self.robot_marker = self.ax.scatter(pos[1], pos[0], c='red', s=100, zorder=5)
                self.canvas.draw()
                self.master.update()
                self.master.after(80)

        else:
            self.ax.set_title("No path found!")
            self.canvas.draw()

    def _on_algorithm_change(self, value):
        self.algorithm_choice = value
        # redraw and animate with new algorithm
        self.draw_background()
        self.animate_path()

if __name__ == "__main__":
    root = tk.Tk()
    gui = RobotGUI(root)
    root.mainloop()
