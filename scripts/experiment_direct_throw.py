#!/usr/bin/env python3
# %%
import numpy as np 
import pandas as pd 
import os 
import random
import torch 
import pickle
import optuna 
import time
import subprocess
import csv
import rospkg

optuna.logging.set_verbosity(optuna.logging.WARNING)
from sklearn.preprocessing import MinMaxScaler

## user inputs ## 
# names and directories 
model_name = "forward_MLP.pth" 
scaler_model_dir_name = "network_data"  
input_scaler_filename = "input_scaler.pkl" 
state_scaler_filename = "state_scaler.pkl"

# neural network configuration  
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

# indices correspond to raw ROS logs 
act_1_idx_ros = 0 
act_2_idx_ros = 1 
act_3_idx_ros = 2 
mid_x_idx_ros = 3 
mid_y_idx_ros = 4 
mid_z_idx_ros = 5 
ee_x_idx_ros = 6 
ee_y_idx_ros = 7 
ee_z_idx_ros = 8 

# initial condittions
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
Q = 1 # landing pose weight term  
n_trials = 1000 # number of trials

device = "cpu"

# setting seeds 
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
set_all_seeds() 

# defining the model 
class MLP_model(torch.nn.Module): 
    def __init__(self, input_flat_size:int, hidden_units:int, output_size:int, num_hidden_layers:int) :
        super().__init__()
        self.input_flat_size = input_flat_size 
        self.hidden_units = hidden_units 
        self.output_size = output_size 
        self.num_hidden_layers = num_hidden_layers 
        hidden_layers = [] 
        in_dimension = self.input_flat_size 
        self.input_layer = torch.nn.Linear(in_features=in_dimension, out_features=self.hidden_units) 
        for i in range(self.num_hidden_layers) : 
            hidden_layers.append(torch.nn.Linear(in_features=self.hidden_units, out_features=self.hidden_units)) 
            hidden_layers.append(torch.nn.ReLU()) 
        self.backbone = torch.nn.Sequential(*hidden_layers) 
        self.output_layer = torch.nn.Linear(in_features=self.hidden_units, out_features=self.output_size) 
        self.relu = torch.nn.ReLU()    
    def forward(self,x): 
        out = self.input_layer(x) 
        out = self.relu(out)
        out = self.backbone(out)  
        out = self.output_layer(out) 
        return out  

# instantiating an NN object 
forward_model = MLP_model(input_flat_size=input_flat_size, hidden_units=hidden_units, output_size=output_size, num_hidden_layers=num_hidden_layers) 

# loading the statedicts 
script_path = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd() 
model_data_path = os.path.join(script_path, scaler_model_dir_name) 
forward_model.load_state_dict(torch.load(os.path.join(model_data_path, model_name), map_location=torch.device('cpu'))) 
forward_model = forward_model.to(device=device) 
forward_model.eval()  

# loading the scalers 
with open(os.path.join(model_data_path, input_scaler_filename), "rb") as file : 
    input_scaler = pickle.load(file) 
with open(os.path.join(model_data_path, state_scaler_filename), "rb") as file : 
    state_scaler = pickle.load(file)

# defining the function to simulate the system given the decision variables
def simulate_sys(u_step: np.array, ramp_steps: int, release_step: int, input_scaler: MinMaxScaler, state_scaler: MinMaxScaler, des_land_pos: np.array) : 
        # configuring the inputs
        ramp_factor = np.linspace(0, 1, ramp_steps) 
        ramp_factor = np.repeat(ramp_factor[:,None], n_inputs, axis=1) 
        u_ramp = ramp_factor * u_step 
        hold_steps = total_steps - ramp_steps 
        if hold_steps > 0 : 
            u_hold = np.tile(u_step, (hold_steps,1)) 
            u_array = np.vstack((u_ramp, u_hold)) 
        else : 
            u_array = u_ramp 

        zeros = np.zeros((max_lag+1, n_inputs)) 
        u_array = np.vstack((zeros, u_array))  

        u_array_scaled = input_scaler.transform(u_array) 
        u_array_torch = torch.from_numpy(u_array_scaled).type(torch.float32) 
        X_init = state_scaler.transform(X) 
        X_init_scaled = torch.from_numpy(X_init).type(torch.float32) 

        current_state = X_init_scaled[max_lag,:] 
        if lag_state == 0 : 
            past_state = X_init_scaled[max_lag:max_lag] 
        else : 
            past_state = X_init_scaled[max_lag-lag_state:max_lag,:] 
            past_state = torch.flatten(input=past_state) 
        current_input = u_array_torch[max_lag,:] 
        if lag_input == 0 : 
            past_input = u_array_torch[max_lag:max_lag,:]
        else : 
            past_input = u_array_torch[max_lag-lag_input:max_lag,:] 
            past_input = torch.flatten(input=past_input) 

        if past_state.size(dim=0) == 0 and past_input.size(dim=0) == 0 : 
            joined_features = torch.concatenate((current_state, current_input), dim=0) 
        elif past_state.size(dim=0) != 0 and past_input.size(dim=0) == 0 : 
            joined_features = torch.concatenate((current_state, past_state, current_input), dim=0)
        elif past_state.size(dim=0) == 0 and past_input.size(dim=0) != 0 : 
            joined_features = torch.concatenate((current_state, current_input, past_input), dim=0) 
        else : 
            joined_features = torch.concatenate((current_state, past_state, current_input, past_input), dim=0)

        preds = [] 
        X_buffer = torch.zeros(len(u_array_torch), n_states)
        for i in range(max_lag+1) : 
            X_buffer[i,:] = X_init_scaled[i,:] 

        with torch.inference_mode(): 
            for i in range(max_lag+1, len(u_array_torch)) : 
                pred = forward_model(joined_features.unsqueeze(0)) 
                pred = pred.squeeze(0) 
                preds.append(pred) 
                X_buffer[i,:] = pred 

                current_state = pred
                if lag_state == 0 : 
                    past_state = X_buffer[i:i] 
                else : 
                    past_state = X_buffer[i-lag_state:i,:] 
                    past_state = torch.flatten(input=past_state) 
                current_input = u_array_torch[i,:] 
                if lag_input == 0 : 
                    past_input = u_array_torch[i:i,:]
                else : 
                    past_input = u_array_torch[i-lag_input:i,:] 
                    past_input = torch.flatten(input=past_input) 

                if past_state.size(dim=0) == 0 and past_input.size(dim=0) == 0 : 
                    joined_features = torch.concatenate((current_state, current_input), dim=0) 
                elif past_state.size(dim=0) != 0 and past_input.size(dim=0) == 0 : 
                    joined_features = torch.concatenate((current_state, past_state, current_input), dim=0)
                elif past_state.size(dim=0) == 0 and past_input.size(dim=0) != 0 : 
                    joined_features = torch.concatenate((current_state, current_input, past_input), dim=0) 
                else : 
                    joined_features = torch.concatenate((current_state, past_state, current_input, past_input), dim=0)
        
        preds_tensor = torch.stack(preds, dim=0) 
        preds_np = preds_tensor.numpy()  
        preds_np = state_scaler.inverse_transform(preds_np) 
        state_traj = np.vstack((X, preds_np))

        diff = np.diff(state_traj, axis=0) 
        velocities = np.vstack([np.zeros((1, n_states)), diff / dt]) 
        delta_z = state_traj[:, -1] - z_g 
        sqrt_term = velocities[:,-1]**2 + (2 * g * delta_z)
        t_flight = (velocities[:,-1] + np.sqrt(sqrt_term)) / g 
        x_landing = state_traj[:,ee_x_idx] + velocities[:,ee_x_idx] * t_flight
        y_landing = state_traj[:,ee_y_idx] + velocities[:,ee_y_idx] * t_flight  

        idx = release_step
        act_landing_x = x_landing[idx] 
        act_landing_y = y_landing[idx]
        land_pos = np.array([act_landing_x, act_landing_y]) 

        dist = np.linalg.norm(des_land_pos - land_pos) 
        return u_array, state_traj, velocities, land_pos, idx, dist

# Function to clear ROS background processes
def kill_ros_processes():
    print("Aggressively cleaning up ROS processes...")
    try:
        subprocess.run(["killall", "-9", "rosmaster", "rosout", "gzserver", "gzclient"], stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        time.sleep(3)
    except Exception as e:
        print(f"Cleanup warning: {e}")

# ====================================================================
# MASTER EXPERIMENT LOOP
# ====================================================================

def main():
    # Load targets
    targets_file = os.path.join(script_path, "workspace_targets.csv")
    targets_df = pd.read_csv(targets_file)
    targets = targets_df.to_numpy()
    
    experiment_results = []
    
    # Get package paths once
    rospack = rospkg.RosPack() 
    pkg_path_sorosimpp = rospack.get_path('sorosimpp_compiled') 
    pkg_path_throw = rospack.get_path('sorosimpp_bayopt_throw') 
    launch_file_path_sorosimpp = pkg_path_sorosimpp + '/launch/sorosimpp_vis.launch' 
    launch_file_path_controller_logger = pkg_path_throw + '/launch/controller_logger.launch' 

    print(f"Starting Master Experiment Loop for {len(targets)} targets.")

    for i, target in enumerate(targets):
        des_land_pos = np.array([target[0], target[1]])
        print(f"\n==================================================")
        print(f"Executing Target {i+1}/{len(targets)}: X={des_land_pos[0]:.4f}, Y={des_land_pos[1]:.4f}")
        print(f"==================================================")

        # 1. OPTUNA OPTIMIZATION (MIL)
        def objective(trial) : 
            u1_step = trial.suggest_float("act1_step", 0, umax)
            u2_step = trial.suggest_float("act2_step", 0, umax) 
            u3_step = trial.suggest_float("act3_step", 0, umax)  
            u_step = np.array([u1_step, u2_step, u3_step]) 
            ramp_steps = trial.suggest_int("ramp_steps", 3, total_steps) 
            release_step = trial.suggest_int("release_step", 0, total_steps + max_lag)
            _, _, _, _, _, dist = simulate_sys(u_step=u_step, ramp_steps=ramp_steps, release_step=release_step, input_scaler=input_scaler, state_scaler=state_scaler, des_land_pos=des_land_pos)    
            return Q * dist

        sampler = optuna.samplers.CmaEsSampler(seed=42)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        print("Running MIL Optimization...")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False) 
        
        best_params = study.best_params
        u_step = np.array([best_params["act1_step"], best_params["act2_step"], best_params["act3_step"]])
        ramp_steps = best_params["ramp_steps"]
        release_step = best_params["release_step"] 

        # Forward pass to get MIL data
        u_data, x_data, velocities, mil_land_pos, release_idx, mil_dist = simulate_sys(u_step=u_step, ramp_steps=ramp_steps, release_step=release_step, input_scaler=input_scaler, state_scaler=state_scaler, des_land_pos=des_land_pos) 
        print(f"MIL Optimization Complete. MIL Error: {mil_dist:.4f}m")

        # Save optimal_inputs.csv for the ROS controller to read
        df_inputs = pd.DataFrame(u_data, columns=["U1", "U2", "U3"]) 
        optimal_inputs_path = os.path.join(script_path, "optimal_inputs.csv")
        df_inputs.to_csv(optimal_inputs_path, index=False) 

        # 2. ROS SIMULATION (Subprocess Architecture)
        kill_ros_processes() # Ensure clean slate

        print("Starting Sorosimpp Launch File...")
        sim_proc = subprocess.Popen(["roslaunch", "sorosimpp_compiled", "sorosimpp_vis.launch"])
        time.sleep(60) # Wait for simulation to stabilize

        print("Starting Controller & Logger...")
        ctrl_proc = subprocess.Popen(["roslaunch", "sorosimpp_bayopt_throw", "controller_logger.launch"])
        time.sleep(30) # Wait for controller to execute trajectory
        time.sleep(5)  # Let it run a bit extra

        print("Shutting down ROS processes...")
        ctrl_proc.terminate()
        time.sleep(5)
        sim_proc.terminate()
        time.sleep(5)
        kill_ros_processes() # Aggressive cleanup

        # 3. POST-PROCESSING LOGS
        print("Processing ROS Simulation Logs...")
        ros_sim_logs_filename = "ROS_sim_logs.csv" 
        ros_sim_logs_path = os.path.join(script_path, ros_sim_logs_filename) 
        
        try:
            df_logs = pd.read_csv(ros_sim_logs_path, header=0)  
            df_logs.set_index('time', inplace=True)
            ros_sim_logs = df_logs.to_numpy() 
            
            # Extract exactly as before
            sim_x_data = ros_sim_logs[:len(x_data), mid_x_idx_ros:] 

            # Exact same math with strict dt
            sim_diff = np.diff(sim_x_data, axis=0) 
            sim_velocities = np.vstack([np.zeros((1, n_states)), sim_diff / dt]) 

            sim_delta_z = sim_x_data[:, -1] - z_g 
            sim_sqrt_term = sim_velocities[:,-1]**2 + (2 * g * sim_delta_z)
            sim_t_flight = (sim_velocities[:,-1] + np.sqrt(sim_sqrt_term)) / g 
            sim_x_landing = sim_x_data[:,ee_x_idx] + sim_velocities[:,ee_x_idx] * sim_t_flight
            sim_y_landing = sim_x_data[:,ee_y_idx] + sim_velocities[:,ee_y_idx] * sim_t_flight 

            sim_act_landing_x = sim_x_landing[release_idx] 
            sim_act_landing_y = sim_y_landing[release_idx]
            sim_land_pos = np.array([sim_act_landing_x, sim_act_landing_y])

            sim_dist = np.linalg.norm(des_land_pos - sim_land_pos)
            print(f"Sim Processing Complete. Sim Error: {sim_dist:.4f}m")
            
        except Exception as e:
            print(f"Error processing logs for target {i+1}: {e}")
            sim_land_pos = np.array([0.0, 0.0])
            sim_dist = 999.0 # Marker for failed log parsing

        # 4. SAVE RESULTS
        experiment_results.append([
            des_land_pos[0], des_land_pos[1],
            mil_land_pos[0], mil_land_pos[1], mil_dist,
            sim_land_pos[0], sim_land_pos[1], sim_dist
        ])

        # Overwrite master CSV periodically so no data is lost if laptop crashes
        results_df = pd.DataFrame(experiment_results, columns=[
            "Target_X", "Target_Y", 
            "MIL_Land_X", "MIL_Land_Y", "MIL_Error", 
            "Sim_Land_X", "Sim_Land_Y", "Sim_Error"
        ])
        results_path = os.path.join(script_path, "final_throwing_experiment.csv")
        results_df.to_csv(results_path, index=False)
        
        # 5. COOL DOWN TIMER
        print(f"Target {i+1} completed. Cooling down for 60 seconds...")
        time.sleep(90)

    print("\n=== EXPERIMENT FULLY COMPLETED ===")
    print(f"All data saved to {results_path}")

if __name__ == '__main__':
    main()