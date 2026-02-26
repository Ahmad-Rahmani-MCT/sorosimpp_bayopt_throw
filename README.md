create a virtual environment
did pip install: 
pip install -r requirements.txt

# SoRoSim++ BayOpt Throw
This package [... insert your description of the package...]

## Install and Build
```bash
git clone git@github.com:Ahmad-Rahmani-MCT/sorosimpp_bayopt_throw.git
```

### SoRoSim++
If you have not the `sorosimpp` package, you can download from [here](https://github.com/Elektron97/sorosimpp_compiled).

### Build
```bash
catkin_make --only-pkg-with-deps sorosimpp_compiled sorosimpp_bayopt_throw -DCMAKE_BUILD_TYPE=Release
```

After creating your virtual environment:
```bash
pip install -r requirements.txt
```
and
```bash
pip install rospakg
```

### Set Robot
Modify the config file in `sorosimpp/config/robot_parameters.yaml` copying this
```yaml
continuum_soft_manipulator:
  link:
    # Geometricals
    length: 0.4         # [m]
    cs_radius: 0.03      # [m]

    # Physicals
    young_modulus: 1.0e+6   # [Pa]
    poisson: 0.5            # [-]
    density: 1.0e+3         # [kg/m^2]
    damping: 1.0e+4         # [Pa * s]
  
  actuation:
    # Actuator 0
    actuator0:
      type: "constant"
      radius: 0.08      # [m]
      phase: 0.0        # [deg]

    # Actuator 1
    actuator1:
      type: "constant"
      radius: 0.08      # [m]
      phase: 120.0      # [deg]
    
    # Actuator 2
    actuator2:
      type: "constant"
      radius: 0.08      # [m]
      phase: 240.0      # [deg]

    ## Uncomment for H-Support ##
    # # Actuator 3
    # actuator3:
    #   type: "helicoidal"
    #   radius: 0.07      # [m]
    #   phase: 0.0        # [deg]
    #   pitch: 1.0        # [m]
    #   sign: "ccw"       # ccw or cw
    
    # # Actuator 4
    # actuator4:
    #   type: "helicoidal"
    #   radius: 0.07      # [m]
    #   phase: 180.0      # [deg]
    #   pitch: 1.0        # [m]
    #   sign: "ccw"       # ccw or cw
    
    # # Actuator 5
    # actuator5:
    #   type: "helicoidal"
    #   radius: 0.07      # [m]
    #   phase: 90.0       # [deg]
    #   pitch: 1.0        # [m]
    #   sign: "cw"        # ccw or cw
    
    # # Actuator 6
    # actuator6:
    #   type: "helicoidal"
    #   radius: 0.07      # [m]
    #   phase: 270.0      # [deg]
    #   pitch: 1.0        # [m]
    #   sign: "cw"        # ccw or cw

  discretization:
    orders:               # order of the modes
      torsion_x:  0
      bending_y:  0
      bending_z:  0
      stretch_x:  0
      shear_y:    0
      shear_z:    0

    gaussian_points: 5    # n° guassian closure points

  g0:                   # Offset SE3
    rotation:
      w: 0.7071
      x: 0.0
      y: 0.7071
      z: 0.0
    translation:
      x: 0.0
      y: 0.0
      z: 0.0
  
  xi_star:              # stress-free strain twist
    torsion_x: 0.0
    bending_y: 0.0
    bending_z: 0.0
    stretch_x: 1.0
    shear_y: 0.0
    shear_z: 0.0

  gravity_twist:         # Gravity acceleration Twist [m/s^2]
    angular:
      x: 0.0
      y: 0.0
      z: 0.0
    linear:
      x: 0.0
      y: 0.0
      z: -9.81
  
  simulation:
    q0:                 [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]          # q(t = 0)
    qdot0:              [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]          # qdot(t = 0)
    solver:             "runge_kutta4"                          # "explicit_euler" | "semi_implicit" | "semi_implicit_damping" | "runge_kutta4"
    sampling_frequency: 1.0e+3                                  # Sampling Frequency [Hz]
    node_frequency:     1.0e+3                                  # Node Frequency [Hz]                               
```

### Run
Create a new launch file inside `sorosimpp/launch/sorosimpp_vis.launch`:
```xml
<launch>
    <!-- Set LD_LIBRARY_PATH for all nodes in this launch file -->
    <env name="LD_LIBRARY_PATH" value="$(find sorosimpp_compiled)/lib:$(env LD_LIBRARY_PATH)" />
    
    <!-- Launch Engine Node -->
    <include file="$(find sorosimpp_compiled)/launch/sorosimpp.launch"/>

    <!-- Launch Visualization Node -->
    <node type="visualization_main" name="sorosimpp_visualization" pkg="sorosimpp_compiled"/>
</launch>
```

Switch to `scripts` dir, by `cd scripts`. Finally,

```bash
python main_direct_throw.py
```
