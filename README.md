# drone_control_pkgs

ROS 2 packages for hexarotor control: a cascaded PD–NMPC outer/inner loop with a
disturbance observer (HGDO or L1 adaptive), driven by RC input.

## Installation

This repository contains three ROS 2 packages: `drone_msgs` (interfaces),
`drone_control` (ament_python), and `drone_dob` (ament_cmake).  The
controller also depends on the external
[`ros2_libcanard_msgs`](https://github.com/kay01-kwon/ros2_libcanard_msgs)
package.

### 1. Create the workspace

```
mkdir -p ~/rotor_sim_ws/src
cd ~/rotor_sim_ws/src
```

### 2. Clone the sources

```
git clone https://github.com/kay01-kwon/drone_control_pkgs.git
git clone https://github.com/kay01-kwon/ros2_libcanard_msgs.git
```

### 3. Install ROS dependencies

```
cd ~/rotor_sim_ws
rosdep install --from-paths src --ignore-src -r -y
```

### 4. Build `drone_msgs` first

`drone_control` and `drone_dob` depend on the interfaces in `drone_msgs`,
so it must be built (and sourced) before the rest.

```
colcon build --packages-select drone_msgs
source install/setup.bash
```

### 5. Build `drone_control` and `drone_dob`

```
colcon build --packages-select drone_control drone_dob --symlink-install
source install/setup.bash
```

Add the source step to your `~/.bashrc` if you launch the stack often:
```
echo "source ~/rotor_sim_ws/install/setup.bash" >> ~/.bashrc
```

## Launch

This stack depends on the `px4_launch` nodes from
[ros2_device_bringup](https://github.com/kay01-kwon/ros2_device_bringup).
Install that package first, then bring the system up in four terminals.

### Terminal 1 — PX4 bridge

```
cd ~/device_ws
source install/setup.bash
ros2 launch px4_launch px4.launch
```

### Terminal 2 — PX4 client

```
cd ~/device_ws
source install/setup.bash
ros2 run px4_launch px4_client_node
```

### Terminal 3 — Disturbance observer

Pick one of the two observers:

```
cd ~/rotor_sim_ws
source install/setup.bash
ros2 launch drone_dob hgdo.launch.py          # HGDO
# or
ros2 launch drone_dob l1_adaptive.launch.py   # L1 adaptive
```

### Terminal 4 — Controller

Launch the controller that matches the observer chosen in Terminal 3:

```
cd ~/rotor_sim_ws
source install/setup.bash
ros2 launch drone_control pd_nmpc_att_with_hgdo.launch.py   # pairs with HGDO
# or
ros2 launch drone_control pd_nmpc_att_with_l1.launch.py     # pairs with L1
```

> **Note**: the controller launch file must match the DOB launched in Terminal 3.

## Activating Manual-Stab Mode

1. **SE switch** — release the kill switch.
2. **SD switch** — move from `DISARMED` to `ARMED`.
3. **SB switch** — leave at the neutral position.
4. Raise throttle to lift off.

<img src="drone_control/figures/Boxer_explanation.png">

## NMPC + DOB Tuning Guide

### 1. HGDO

```
dob:
  dob_looptime: 0.010
  eps_f:   0.10
  eps_tau: 0.15
```

### 2. L1 adaptive

```
l1_adaptive:
  dob_looptime: 0.010
  As_array: [-15.0, -15.0, -15.0,   # vx, vy, vz (body frame)
             -15.0, -15.0, -15.0]   # wx, wy, wz (body frame)
  freq_cutoff_trans: 2.0
  freq_cutoff_rot:   2.0
```

## To-do List

### 1. DOB implementation
- [x] HGDO
- [x] L1 adaptive
<!-- - [ ] UKF/EKF for DOB -->

### 2. Control implementation
- [x] Manual control (velocity-mode fallback for emergencies)
- [x] Simple NMPC
- [x] NMPC (acados) with DOB
- [x] NMPC with moment feedforward
- [x] NMPC / RC integration with DOB
