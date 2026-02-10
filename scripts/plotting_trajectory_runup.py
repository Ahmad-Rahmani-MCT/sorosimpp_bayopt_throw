#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# =========================== CONFIGURATION =========================== #
REQUIRED_FRAMES = ["base"] + [f"cs{i}" for i in range(1, 39)] + ["tip"]

# --- THROW SETTINGS ---
RELEASE_INDEX = 12

# --- GOAL SETTINGS ---
GOAL_POS = [0.18, -0.18, -1.0]

# --- ENVIRONMENT SETTINGS ---
GROUND_Z = -1.0
GRAVITY = 9.8

# --- VISUALIZATION SETTINGS ---
WORKSPACE_RADIUS = 0.18
WORKSPACE_HEIGHT_START = 0.0
WORKSPACE_HEIGHT_END = -1.0
WORKSPACE_ALPHA = 0.05
BALL_COLOR = 'red'

# --- DATA / TIMING ---
DATA_SAMPLING_FREQ = 10  # Hz
FPS = 20
# ===================================================================== #


# =========================== PROJECTILE =========================== #
class Projectile:
    def __init__(self, start_pos, start_vel, start_idx):
        self.start_pos = np.array(start_pos)
        self.start_vel = np.array(start_vel)
        self.start_idx = start_idx

        self.calculate_trajectory()

    def calculate_trajectory(self):
        # --- ANALYTICAL SOLUTION ---
        x0, y0, z0 = self.start_pos
        vx0, vy0, vz0 = self.start_vel

        delta_z = z0 - GROUND_Z
        sqrt_term = np.sqrt(vz0**2 + 2 * GRAVITY * delta_z)
        self.t_flight = (vz0 + sqrt_term) / GRAVITY

        # Exact landing position
        land_x = x0 + vx0 * self.t_flight
        land_y = y0 + vy0 * self.t_flight
        self.land_pos = np.array([land_x, land_y, GROUND_Z])

        # Smooth trajectory for plotting
        t = np.linspace(0, self.t_flight, 50)
        traj_x = x0 + vx0 * t
        traj_y = y0 + vy0 * t
        traj_z = z0 + vz0 * t - 0.5 * GRAVITY * t**2
        self.traj = np.column_stack((traj_x, traj_y, traj_z))

    def get_position_at_frame(self, frame_idx):
        frames_since_release = frame_idx - self.start_idx
        if frames_since_release < 0:
            return None

        dt = 1.0 / DATA_SAMPLING_FREQ
        t = frames_since_release * dt

        z = self.start_pos[2] + self.start_vel[2] * t - 0.5 * GRAVITY * t**2
        
        if z <= GROUND_Z or t >= self.t_flight:
            return self.land_pos

        x = self.start_pos[0] + self.start_vel[0] * t
        y = self.start_pos[1] + self.start_vel[1] * t
        return np.array([x, y, z])


# =========================== DATA LOADING =========================== #
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "logged_data_csv", "ros_data_logged.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(file_path)

    df = pd.read_csv(file_path)
    return df, script_dir


# =========================== VELOCITY =========================== #
def calculate_velocities(df):
    dt = 1.0 / DATA_SAMPLING_FREQ
    xs = df['tip_pos_x'].values
    ys = df['tip_pos_y'].values
    zs = df['tip_pos_z'].values

    positions = np.column_stack((xs, ys, zs))
    diff = np.diff(positions, axis=0)

    velocities = np.vstack([
        np.zeros((1, 3)),
        diff / dt
    ])
    return velocities


# =========================== PLOTTING HELPERS =========================== #
def setup_axes(ax):
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(GROUND_Z, 0.0)
    ax.grid(False)


def plot_environment(ax):
    z = np.linspace(WORKSPACE_HEIGHT_START, WORKSPACE_HEIGHT_END, 30)
    theta = np.linspace(0, 2 * np.pi, 30)
    theta_grid, z_grid = np.meshgrid(theta, z)
    x_grid = WORKSPACE_RADIUS * np.cos(theta_grid)
    y_grid = WORKSPACE_RADIUS * np.sin(theta_grid)
    ax.plot_surface(x_grid, y_grid, z_grid, alpha=WORKSPACE_ALPHA, color='blue')

    xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 10), np.linspace(-0.5, 0.5, 10))
    zz = np.full_like(xx, GROUND_Z)
    ax.plot_surface(xx, yy, zz, alpha=0.1, color='green')
    
    # Plot Goal
    ax.scatter(GOAL_POS[0], GOAL_POS[1], GOAL_POS[2], color='green', s=150, marker='X', label="Goal")


# =========================== STATIC PLOT =========================== #
def generate_static_plot(df, projectile, dist_error, script_dir):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f"Throw Trajectory (Analytical)\nFinal Error: {dist_error:.4f} m")
    setup_axes(ax)
    plot_environment(ax)

    # --- PLOT GHOST LINES ---
    # Step = 1 (Every frame) and Alpha = 0.5 (Darker)
    for i in range(0, RELEASE_INDEX + 1, 1):
        xs = [df.iloc[i][f'{f}_pos_x'] for f in REQUIRED_FRAMES]
        ys = [df.iloc[i][f'{f}_pos_y'] for f in REQUIRED_FRAMES]
        zs = [df.iloc[i][f'{f}_pos_z'] for f in REQUIRED_FRAMES]
        ax.plot(xs, ys, zs, color='gray', alpha=0.5, linewidth=1)

    # Robot pose at release (Solid Black)
    xs = [df.iloc[RELEASE_INDEX][f'{f}_pos_x'] for f in REQUIRED_FRAMES]
    ys = [df.iloc[RELEASE_INDEX][f'{f}_pos_y'] for f in REQUIRED_FRAMES]
    zs = [df.iloc[RELEASE_INDEX][f'{f}_pos_z'] for f in REQUIRED_FRAMES]
    ax.plot(xs, ys, zs, color='black', linewidth=2.5, label="Robot at Release")

    # Projectile
    ax.plot(projectile.traj[:, 0],
            projectile.traj[:, 1],
            projectile.traj[:, 2],
            '--', color=BALL_COLOR, label="Projectile Path")

    # Landing
    ax.scatter(*projectile.land_pos, color='red', s=100, marker='X', label="Actual Landing")

    # Annotation
    land_x, land_y, land_z = projectile.land_pos
    ax.text(land_x, land_y, land_z + 0.1, f"({land_x:.2f}, {land_y:.2f})", color='red')

    ax.legend(loc='upper right')
    
    out_dir = os.path.join(script_dir, "throw_plots_runup")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{GOAL_POS[0]}_{GOAL_POS[1]}_{GOAL_POS[2]}.png"
    plt.savefig(os.path.join(out_dir, filename), dpi=300)
    plt.close(fig)
    print(f"Static plot saved to {filename}")


# =========================== ANIMATION =========================== #
def generate_animation(df, projectile, dist_error, script_dir):
    total_frames = len(df) + 30
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    setup_axes(ax)
    plot_environment(ax)

    robot_line, = ax.plot([], [], [], color='black', linewidth=2, label="Robot")
    ball_point, = ax.plot([], [], [], 'o', color=BALL_COLOR, label="Ball")
    
    ax.legend(loc='upper right')

    def update(frame):
        if frame < len(df):
            xs = [df.iloc[frame][f'{f}_pos_x'] for f in REQUIRED_FRAMES]
            ys = [df.iloc[frame][f'{f}_pos_y'] for f in REQUIRED_FRAMES]
            zs = [df.iloc[frame][f'{f}_pos_z'] for f in REQUIRED_FRAMES]
            robot_line.set_data(xs, ys)
            robot_line.set_3d_properties(zs)
            tip = np.array([xs[-1], ys[-1], zs[-1]])
        else:
            tip = projectile.land_pos 

        if frame < RELEASE_INDEX:
            ball = tip
            status = "ATTACHED"
        else:
            ball = projectile.get_position_at_frame(frame)
            status = "THROWN"

        ax.set_title(f"Status: {status}\nFinal Error: {dist_error:.4f} m")

        ball_point.set_data([ball[0]], [ball[1]])
        ball_point.set_3d_properties([ball[2]])
        
        return robot_line, ball_point

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000/FPS)
    
    out_dir = os.path.join(script_dir, "throw_animations_runup")
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{GOAL_POS[0]}_{GOAL_POS[1]}_{GOAL_POS[2]}.mp4"
    ani.save(os.path.join(out_dir, filename), writer=animation.FFMpegWriter(fps=FPS))
    plt.close(fig)
    print(f"Animation saved to {filename}")


# =========================== MAIN =========================== #
def main():
    df, script_dir = load_data()
    velocities = calculate_velocities(df)

    rel_pos = [
        df.iloc[RELEASE_INDEX]['tip_pos_x'],
        df.iloc[RELEASE_INDEX]['tip_pos_y'],
        df.iloc[RELEASE_INDEX]['tip_pos_z']
    ]
    rel_vel = velocities[RELEASE_INDEX]

    print("Release index:", RELEASE_INDEX)
    print("Release position:", rel_pos)
    print("Release velocity (OPTIMIZATION-STYLE):", rel_vel)

    projectile = Projectile(rel_pos, rel_vel, RELEASE_INDEX)
    
    # Calculate Error
    land_xy = projectile.land_pos[:2]
    goal_xy = np.array(GOAL_POS[:2])
    dist_error = np.linalg.norm(land_xy - goal_xy)
    
    print(f"Calculated Landing: {land_xy}")
    print(f"Goal: {goal_xy}")
    print(f"Error: {dist_error}")

    generate_static_plot(df, projectile, dist_error, script_dir)
    generate_animation(df, projectile, dist_error, script_dir)

    print("Done.")


if __name__ == "__main__":
    main()