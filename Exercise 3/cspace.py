import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# -------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------
def torus_coords(theta1, theta2, R=0.75, r=0.25):
    X = (R + r * np.cos(theta2)) * np.cos(theta1)
    Y = (R + r * np.cos(theta2)) * np.sin(theta1)
    Z = r * np.sin(theta2)
    return X, Y, Z

def cylinder_coords(theta, d, R=2.0):
    X = R * np.cos(theta)
    Y = R * np.sin(theta)
    Z = d
    return X, Y, Z

def two_link_positions(theta1, theta2, L1=1.0, L2=0.6):
    x1 = L1 * np.cos(theta1)
    y1 = L1 * np.sin(theta1)
    x2 = x1 + L2 * np.cos(theta1 + theta2)
    y2 = y1 + L2 * np.sin(theta1 + theta2)
    return (0, x1, x2), (0, y1, y2)

def revolute_prismatic_positions(theta, d, L1=1.0):
    x1 = L1 * np.cos(theta)
    y1 = L1 * np.sin(theta)
    x2 = x1 + d * np.cos(theta)
    y2 = y1 + d * np.sin(theta)
    return (0, x1, x2), (0, y1, y2)

def mobile_robot_shape(x, y, theta):
    # Rectangle centered at (x, y)
    body = np.array([
        [0.2, -0.1],
        [0.2,  0.1],
        [-0.2,  0.1],
        [-0.2, -0.1],
        [0.2, -0.1]
    ])
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])
    rotated = (R @ body.T).T + np.array([x, y])
    return rotated[:, 0], rotated[:, 1]

# -------------------------------------------------------------
# Setup figure
# -------------------------------------------------------------
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')
plt.subplots_adjust(bottom=0.35, wspace=0.3)

# Workspace setup
robot_line, = ax1.plot([], [], 'o-', lw=4, color='royalblue')
orient_line, = ax1.plot([], [], 'k-', lw=2)  # persistent orientation arrow
ax1.set_xlim(-2, 2)
ax1.set_ylim(-2, 2)
ax1.set_aspect('equal')
ax1.grid(True, alpha=0.3)
ax1.set_title("Robot Workspace")

# C-space setup
ax2.set_title("Configuration Space")
ax2.set_box_aspect([1, 1, 1])
ax2.set_axis_off()
point, = ax2.plot([], [], [], 'ro', markersize=8)
surf = None

# -------------------------------------------------------------
# Sliders
# -------------------------------------------------------------
ax_theta1 = plt.axes([0.15, 0.22, 0.65, 0.03])
ax_theta2 = plt.axes([0.15, 0.17, 0.65, 0.03])
slider1 = Slider(ax_theta1, 'Joint 1 / X / θ', -2, 2, valinit=0)
slider2 = Slider(ax_theta2, 'Joint 2 / D / θ', -np.pi, np.pi, valinit=0)

# Y-slider for mobile robot only
ax_y = plt.axes([0.15, 0.12, 0.65, 0.03])
slider_y = Slider(ax_y, 'Y', -2, 2, valinit=0)

# -------------------------------------------------------------
# Buttons
# -------------------------------------------------------------
ax_button1 = plt.axes([0.15, 0.02, 0.2, 0.05])
ax_button2 = plt.axes([0.4, 0.02, 0.2, 0.05])
ax_button3 = plt.axes([0.65, 0.02, 0.2, 0.05])

btn_torus = Button(ax_button1, "2-Link (Torus)", color='lightgray', hovercolor='lightblue')
btn_cylinder = Button(ax_button2, "Revolute+Prismatic", color='lightgray', hovercolor='lightblue')
btn_mobile = Button(ax_button3, "Mobile Robot", color='lightgray', hovercolor='lightgreen')

# -------------------------------------------------------------
# Global variables
# -------------------------------------------------------------
mode = "torus"

# -------------------------------------------------------------
# Update function
# -------------------------------------------------------------
def update(val):
    global mode
    t1, t2 = slider1.val, slider2.val
    ax1.set_title(f"Robot Workspace ({mode})")

    if mode == "torus":
        x, y = two_link_positions(t1, t2)
        robot_line.set_data(x, y)
        Xp, Yp, Zp = torus_coords(t1, t2)
        point.set_data([Xp], [Yp])
        point.set_3d_properties([Zp])
        orient_line.set_data([], [])

    elif mode == "cylinder":
        d = t2
        x, y = revolute_prismatic_positions(t1, d)
        robot_line.set_data(x, y)
        Xp, Yp, Zp = cylinder_coords(t1, d)
        point.set_data([Xp], [Yp])
        point.set_3d_properties([Zp])
        orient_line.set_data([], [])

    elif mode == "mobile":
        x = slider1.val
        y = slider_y.val
        theta = t2

        # Update robot body
        Xb, Yb = mobile_robot_shape(x, y, theta)
        robot_line.set_data(Xb, Yb)

        # Update fixed orientation arrow
        L = 0.4
        orient_line.set_data([x, x + L*np.cos(theta)], [y, y + L*np.sin(theta)])

        # Red dot in C-space
        Xp, Yp, Zp = x, y, 0.2 * np.sin(theta)
        point.set_data([Xp], [Yp])
        point.set_3d_properties([Zp])

    fig.canvas.draw_idle()

# -------------------------------------------------------------
# Surface helper
# -------------------------------------------------------------
def set_surface(X, Y, Z, title, color='deepskyblue'):
    global surf, point
    ax2.clear()
    surf = ax2.plot_surface(X, Y, Z, color=color, alpha=0.6, edgecolor='none')
    ax2.set_title(title)
    ax2.set_box_aspect([1, 1, 0.5])
    ax2.set_axis_off()
    point, = ax2.plot([], [], [], 'ro', markersize=8)

# -------------------------------------------------------------
# Mode switchers
# -------------------------------------------------------------
def set_torus(event):
    global mode
    mode = "torus"
    U, V = np.meshgrid(np.linspace(0, 2*np.pi, 40), np.linspace(0, 2*np.pi, 20))
    X, Y, Z = torus_coords(U, V)
    set_surface(X, Y, Z, "C-Space = S¹ × S¹ (Torus)", color='deepskyblue')
    slider1.set_val(0)
    slider2.set_val(0)
    slider_y.ax.set_visible(False)
    update(None)

def set_cylinder(event):
    global mode
    mode = "cylinder"

    # Cylinder surface mesh
    U, V = np.meshgrid(np.linspace(0, 2*np.pi, 40), np.linspace(-1.5, 1.5, 20))
    X, Y, Z = cylinder_coords(U, V)
    set_surface(X, Y, Z, "C-Space = S¹ × ℝ (Cylinder)", color='skyblue')

    # Reset sliders
    slider1.set_val(0)

    # Adjust slider2 to match cylinder height
    slider2.valmin = -1.5
    slider2.valmax = 1.5
    slider2.ax.set_xlim(slider2.valmin, slider2.valmax)  # Update slider axis
    slider2.set_val(0)

    # Hide Y-slider (not needed)
    slider_y.ax.set_visible(False)

    # Update plots
    update(None)

def set_mobile(event):
    global mode
    mode = "mobile"
    U, V = np.meshgrid(np.linspace(-2, 2, 40), np.linspace(-2, 2, 20))
    Z = 0.0 * U
    set_surface(U, V, Z, "C-Space ≈ ℝ² × S¹ (Plane × Circle)", color='lightgreen')
    slider1.set_val(0)
    slider2.set_val(0)
    slider_y.set_val(0)
    slider_y.ax.set_visible(True)
    update(None)

# -------------------------------------------------------------
# Connect controls
# -------------------------------------------------------------
btn_torus.on_clicked(set_torus)
btn_cylinder.on_clicked(set_cylinder)
btn_mobile.on_clicked(set_mobile)
# Throttle rapid slider events using a short timer so dragging is smoother
slider_timer = None

def _perform_scheduled_update():
    global slider_timer
    try:
        update(None)
    finally:
        try:
            if slider_timer is not None:
                slider_timer.stop()
        except Exception:
            pass
        slider_timer = None


def _schedule_update(val):
    global slider_timer
    # stop any existing timer
    try:
        if slider_timer is not None:
            slider_timer.stop()
    except Exception:
        pass
    try:
        # If there's no active timer, perform an immediate update so the UI
        # responds at once while we still throttle subsequent rapid events.
        if slider_timer is None:
            try:
                update(None)
            except Exception:
                pass

        # small interval for snappy updates while dragging
        slider_timer = fig.canvas.new_timer(interval=10)
        slider_timer.single_shot = True
        slider_timer.add_callback(_perform_scheduled_update)
        slider_timer.start()
    except Exception:
        # fallback to immediate update
        update(None)


# Use throttled callbacks for sliders
slider1.on_changed(_schedule_update)
slider2.on_changed(_schedule_update)
slider_y.on_changed(_schedule_update)

# Ensure sliders generate events while being dragged and add
# mouse handlers so the first click activates dragging immediately
# (avoids needing a double-click to start interacting on some platforms).
slider1.eventson = True
slider2.eventson = True
slider_y.eventson = True

# Track active mouse-drag on slider axes and forward motion events
# to the throttled update routine. This makes the UI respond
# continuously while the user drags the slider handle.
_mouse_pressed = False
_active_slider_axes = None

def _on_mouse_press(event):
    global _mouse_pressed, _active_slider_axes
    # Only start tracking for left-button presses inside one of the
    # slider axes. This avoids interfering with other interactions.
    if event.button == 1 and event.inaxes in (ax_theta1, ax_theta2, ax_y):
        _mouse_pressed = True
        _active_slider_axes = event.inaxes
        # immediate visual feedback
        _schedule_update(None)

def _on_mouse_release(event):
    global _mouse_pressed, _active_slider_axes
    if _mouse_pressed:
        _mouse_pressed = False
        _active_slider_axes = None
        # final update on release
        _schedule_update(None)

def _on_mouse_move(event):
    # While the left button is held and the cursor moves inside the
    # active slider axis, trigger the throttled update so the slider
    # responds smoothly to dragging.
    if _mouse_pressed and event.inaxes is _active_slider_axes:
        _schedule_update(None)

# Connect canvas events
fig.canvas.mpl_connect('button_press_event', _on_mouse_press)
fig.canvas.mpl_connect('button_release_event', _on_mouse_release)
fig.canvas.mpl_connect('motion_notify_event', _on_mouse_move)

# -------------------------------------------------------------
# Start with torus
# -------------------------------------------------------------
set_torus(None)
plt.show()
