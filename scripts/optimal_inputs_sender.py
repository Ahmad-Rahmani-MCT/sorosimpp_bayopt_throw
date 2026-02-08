import roslaunch 
import rospy 
import rospkg
import time 
import subprocess # Added to run terminal commands
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os 
import random
import torch 
import pickle
import optuna 
optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.preprocessing import MinMaxScaler 

script_path = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd() 
dt = 0.1 
z_g = -1
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
des_land_pos = [0.15, 0.15] # desired landing pose 
g = 9.8 # gravity acceleration


# closing any active nodes (to prevent errors)
print("closing any active nods")
try:
    # 'rosnode kill -a' kills all nodes currently registered with the master
    # supress output to keep console clean
    subprocess.call(["rosnode", "kill", "-a"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # time to actually close
    time.sleep(2) 
except Exception as e:
    print("warning during closing (master might not be running yet):", e)

# finding the ros packages dynamically
rospack = rospkg.RosPack() 
pkg_path_sorosimpp = rospack.get_path('sorosimpp_compiled') 
pkg_path_throw = rospack.get_path('sorosimpp_bayopt_throw') 
launch_file_path_sorosimpp = pkg_path_sorosimpp + '/launch/sorosimpp_vis.launch' 
launch_file_path_controller_logger = pkg_path_throw + '/launch/controller_logger.launch' 

# configuring the launch parent 
uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
roslaunch.configure_logging(uuid)

# creating the launch parent object 
launch_sorosimpp = roslaunch.parent.ROSLaunchParent(uuid, [launch_file_path_sorosimpp]) 
launch_controller_logger = roslaunch.parent.ROSLaunchParent(uuid, [launch_file_path_controller_logger]) 

# starting the launch files 
print("starting the sorosimpp launch file")
launch_sorosimpp.start() 
rospy.sleep(10) 
print("starting the controller and logger launch file") 
launch_controller_logger.start() 

# running for a while 
try:
    rospy.sleep(5) 
except rospy.ROSInterruptException:
    pass

# stopping the launch files 
print("shutting down the controller and logger launch file")
launch_controller_logger.shutdown() 
rospy.sleep(3) 
launch_sorosimpp.shutdown()


# %% [markdown]
# visualizing the ROS simulation results

# %%
# reading the csv file, setting time to index and conversion to numpy arrays
ros_sim_logs_filename = "ROS_sim_logs.csv" 
ros_sim_logs_path = os.path.join(script_path, ros_sim_logs_filename) 
df = pd.read_csv(ros_sim_logs_path, header=0)  
df.set_index('time', inplace=True)
ros_sim_logs = df.to_numpy() 
print("ROS simulation logs shape: ", ros_sim_logs.shape) 

# extracting the inputs and states
u_data = ros_sim_logs[:32, 0:3]
x_data = ros_sim_logs[:32, 3:] 

# calculating the velocties 
diff = np.diff(x_data, axis=0) 
velocities = np.vstack([np.zeros((1, 6)), diff / dt])

# predicting the landing positions 
delta_z = x_data[:, -1] - z_g 
sqrt_term = velocities[:,-1]**2 + (2 * g * delta_z)
t_flight = (velocities[:,-1] + np.sqrt(sqrt_term)) / g 
x_landing = x_data[:,ee_x_idx] + velocities[:,ee_x_idx] * t_flight
y_landing = x_data[:,ee_y_idx] + velocities[:,ee_y_idx] * t_flight 

# final landing based on the release time 
release_idx = 10
idx = release_idx
act_landing_x = x_landing[idx] 
act_landing_y = y_landing[idx]
final_land_pos = np.array([act_landing_x, act_landing_y])

# Time vectors for plotting
time_steps = np.arange(len(x_data)) * dt
time_inputs = np.arange(len(u_data)) * dt 

script_directory = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd() 
runup_throw_performance_dirname = "runup_throw_performance" 
runup_throw_performance_path = os.path.join(script_directory, runup_throw_performance_dirname)
os.makedirs(runup_throw_performance_path, exist_ok=True) 

# creating the folder to save the plots 
sim_result_dirname = "sim_results" 
sim_result_path = os.path.join(runup_throw_performance_path, sim_result_dirname)
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


