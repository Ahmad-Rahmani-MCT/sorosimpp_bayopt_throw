import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os 

script_path = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd() 
direct_throw_performance_dirname = "direct_throw_performance" 
direct_throw_performance_path = os.path.join(script_path, direct_throw_performance_dirname)
os.makedirs(direct_throw_performance_path, exist_ok=True) 

act_1_idx_ros = 0 
act_2_idx_ros = 1 
act_3_idx_ros = 2 
mid_x_idx_ros = 3 
mid_y_idx_ros = 4 
mid_z_idx_ros = 5 
ee_x_idx_ros = 6 
ee_y_idx_ros = 7 
ee_z_idx_ros = 8  

lag_input = 0 
lag_state = 1 
max_lag = max(lag_state, lag_state)
num_hidden_layers = 0
hidden_units = 30 
input_flat_size = 6 + (lag_state*6) + 3 + (lag_input*3) 
output_size = 6 
n_states = 6 
n_inputs = 3 
mid_x_idx = 0 
mid_y_idx = 1 
mid_z_idx = 2 
ee_x_idx = 3 
ee_y_idx = 4 
ee_z_idx = 5  

x_intial = np.array([3.3065416622541037e-06, 0, -0.19036912150652113, 6.0826336879046396e-06, 0, -0.3907576704717413])
X = np.tile(x_intial, (max_lag+1,1)) # handling the state lags
u_initial = np.array([0,0,0]) 
U = np.tile(u_initial, (max_lag+1, 1))  # handling the input lags 
# optimization parameters
umax = 12 # max input 
dumax = 12 # max input change (from dataset)
tmax = 3  # max simulation time
dt = 0.1 # sampling time 
total_steps = int(tmax/dt) 
z_g = -1 # structure height 
g = 9.8 # gravity acceleration
des_land_pos = [0.12, -0.11] # desired landing pose 
Q = 1 # landing pose weight term  
n_trials = 1000 # number of trials  

direct_throw_performance_path = os.path.join(script_path, direct_throw_performance_dirname)
os.makedirs(direct_throw_performance_path, exist_ok=True) 






release_idx = 5





# reading the csv file, setting time to index and conversion to numpy arrays
ros_sim_logs_filename = "ROS_sim_logs.csv" 
ros_sim_logs_path = os.path.join(script_path, ros_sim_logs_filename) 
df = pd.read_csv(ros_sim_logs_path, header=0)  
df.set_index('time', inplace=True)
ros_sim_logs = df.to_numpy() 
print("ROS simulation logs shape: ", ros_sim_logs.shape) 

# extracting the inputs and states
u_data = ros_sim_logs[:32, act_1_idx_ros:mid_x_idx_ros]
x_data = ros_sim_logs[:32, mid_x_idx_ros:] 

# calculating the velocties 
diff = np.diff(x_data, axis=0) 
velocities = np.vstack([np.zeros((1, n_states)), diff / dt]) 

# predicting the landing positions 
delta_z = x_data[:, -1] - z_g 
sqrt_term = velocities[:,-1]**2 + (2 * g * delta_z)
t_flight = (velocities[:,-1] + np.sqrt(sqrt_term)) / g 
x_landing = x_data[:,ee_x_idx] + velocities[:,ee_x_idx] * t_flight
y_landing = x_data[:,ee_y_idx] + velocities[:,ee_y_idx] * t_flight 

# final landing based on the release time 
idx = release_idx
act_landing_x = x_landing[idx] 
act_landing_y = y_landing[idx]
final_land_pos = np.array([act_landing_x, act_landing_y])

# Time vectors for plotting
time_steps = np.arange(len(x_data)) * dt
time_inputs = np.arange(len(u_data)) * dt

# creating the folder to save the plots 
sim_result_dirname = f"sim_results_{des_land_pos[0]}_{des_land_pos[1]}" 
sim_result_path = os.path.join(direct_throw_performance_path, sim_result_dirname)
os.makedirs(sim_result_path, exist_ok=True) 

# plot 1: Actuation Profiles
plot_name = "act_inputs_sim.png"
plt.figure(figsize=(10, 5))
plt.plot(time_inputs, u_data[:, 0], label='Actuator 1', linewidth=2)
plt.plot(time_inputs, u_data[:, 1], label='Actuator 2', linewidth=2)
plt.plot(time_inputs, u_data[:, 2], label='Actuator 3', linewidth=2)
# Add vertical line for release time
# Note: release_idx corresponds to preds, we need to shift it for inputs which has lag padding
max_lag = max(lag_input, lag_state)
release_time_plot_u = (release_idx - 1) * dt # IN THE U DOMAIN
release_time_plot = (release_idx) * dt # IN THE X DOMAIN
plt.axvline(x=release_time_plot_u, color='k', linestyle='--', label=f'Actuator Input Corresponding to Release ({release_time_plot_u:.2f}s)')
plt.axvline(x=release_time_plot, color='k', linestyle='--', label=f'Release Instance ({release_time_plot:.2f}s)')
plt.title('Optimal Input Profiles')
plt.xlabel('Time (s)')
plt.ylabel('Actuation Input')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(sim_result_path, plot_name), dpi=300, bbox_inches="tight")
plt.close()

# Plot 2: 2D End Effector Trajectory (X-Y)
plot_name = "2D_EE_traj_sim.png"
plt.figure(figsize=(8, 8))
plt.plot(x_data[:, ee_x_idx], x_data[:, ee_y_idx], 'b-', linewidth=2, label='Trajectory')
# Mark start
plt.plot(x_data[0, ee_x_idx], x_data[0, ee_y_idx], 'go', label='Start') 
# Mark end 
plt.plot(x_data[-1, ee_x_idx], x_data[-1, ee_y_idx], 'ko', label='End') 
# Mark release point
plt.plot(x_data[release_idx, ee_x_idx], x_data[release_idx, ee_y_idx], 'ro', label='Release Point')
# Mark landing point and desired landing point 
plt.plot(final_land_pos[0], final_land_pos[1], 'rx', label='Landing Point')
plt.plot(des_land_pos[0], des_land_pos[1], 'kx', label='Desired Landing Point')
plt.title('End Effector Trajectory (X-Y)')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()
plt.axis('equal')
plt.grid(True) 
cx, cy = x_data[0, ee_x_idx], x_data[0, ee_y_idx] 
half = 0.2 # how far from the center 
ax = plt.gca()
ax.set_xlim(cx - half, cx + half)
ax.set_ylim(cy - half, cy + half)
ax.set_aspect('equal', adjustable='box')  # optional: square scaling
plt.tight_layout() # adjust padding so labels and titles dont get clipped in the saved file 
plt.savefig(os.path.join(sim_result_path, plot_name), dpi=300, bbox_inches="tight")
plt.close() 

# Plot 3: Landing Targets
plot_name = "Landing_sim.png"
plt.figure(figsize=(8, 8))
# Plot Desired
plt.scatter(des_land_pos[0], des_land_pos[1], color='green', s=200, marker='X', label='Desired Landing')
# Plot Actual
plt.scatter(final_land_pos[0], final_land_pos[1], color='red', s=150, marker='o', label='Predicted Landing')
# Draw line connecting them
plt.plot([des_land_pos[0], final_land_pos[0]], [des_land_pos[1], final_land_pos[1]], 'k--', alpha=0.5)

# Add circles for visual context (distance error)
error = np.linalg.norm(final_land_pos - des_land_pos)
circle = plt.Circle(des_land_pos, error, color='gray', fill=False, linestyle=':', alpha=0.5)
plt.gca().add_patch(circle)

plt.title(f'Landing Accuracy\nError: {error:.4f} m')
plt.xlabel('X Landing (m)')
plt.ylabel('Y Landing (m)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.tight_layout() # adjust padding so labels and titles dont get clipped in the saved file 
plt.savefig(os.path.join(sim_result_path, plot_name), dpi=300, bbox_inches="tight")
plt.close()

# plot 4: Velocity profile 
plot_name = "vel_profile_sim.png"
plt.figure(figsize=(10, 5))
plt.plot(time_steps, velocities[:, ee_x_idx], label='End Effector X Velocity', linewidth=2)
plt.plot(time_steps, velocities[:, ee_y_idx], label='End Effector Y Velocity', linewidth=2)
plt.plot(time_steps, velocities[:, ee_z_idx], label='End Effector Z velocity', linewidth=2)
# Add vertical line for release time
# Note: release_idx corresponds to preds, we need to shift it for inputs which has lag padding
release_time_plot_v = (release_idx) * dt 
plt.axvline(x=release_time_plot_v, color='k', linestyle='--', label=f'Release Instance ({release_time_plot_v:.2f}s)')
plt.title('Velocity Profiles')
plt.xlabel('Time (s)')
plt.ylabel('Velocity [m/s]')
plt.legend()
plt.grid(True)
plt.tight_layout() # adjust padding so labels and titles dont get clipped in the saved file 
plt.savefig(os.path.join(sim_result_path, plot_name), dpi=300, bbox_inches="tight")
plt.close()

# Plot 5: Absolute Velocity Profile 
plot_name = "abs_vel_profile_sim.png"
ee_vel_indices = [ee_x_idx, ee_y_idx, ee_z_idx]
v_abs = np.linalg.norm(velocities[:, ee_vel_indices], axis=1) 
plt.figure(figsize=(10, 5))
plt.plot(time_steps, v_abs, label='End Effector Absolute Velocity', linewidth=2, color='purple')
# Add vertical line for release time
# Note: release_idx corresponds to preds, we need to shift it for inputs which has lag padding
release_time_plot_v = (release_idx) * dt 
plt.axvline(x=release_time_plot_v, color='k', linestyle='--', label=f'Release Instance ({release_time_plot_v:.2f}s)')
plt.title('Absolute Velocity (Magnitude) Profile')
plt.xlabel('Time (s)')
plt.ylabel('Velocity Magnitude [m/s]')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(sim_result_path, plot_name), dpi=300, bbox_inches="tight")
plt.close()