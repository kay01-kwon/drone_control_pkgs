# drone_control_pkgs

## Installation

```
cd ~/rotor_sim_ws/
```
Install drone_control pacakge
```
colcon build --packages-select drone_control --symlink-install
```

Install drone_dob package
```
colcon build --packages-select drone_dob --symlink-install
```


## Run Launch

Go to the following link and then install the pacagkes.

https://github.com/kay01-kwon/ros2_device_bringup

After install the pacakge, run the node with px6x mini.

Terminal 1

Navigate to the workspace.
```
cd ~/device_ws 
```
Source setup.bash.
```
source install/setup.bash
```
Run the px4 node.
```
ros2 launch px4_launch px4.launch
```
Terminal 2
```
cd ~/device_ws
```

```
source install/setup.bash
```

```
ros2 run px4_launch px4_client_node
```

Terminal3
```
cd ~/rotor_sim_ws
```

```
source install/setup.bash
```

```
ros2 launch drone_dob hgdo.launch.py
```


Terminal 4
```
cd ~/rotor_sim_ws
```

```
source install/setup.bash
```

```
ros2 launch drone_control rc_control.launch.py
```