# Command Filtered Backstepping for Hexacopter Position Control

## 1. Motivation

In the current PD+NMPC control architecture for the S550 hexacopter, a limit cycle oscillation of approximately 0.3--0.5 Hz with ~7 cm amplitude is observed during hover in hardware, but not in simulation. Root cause analysis reveals the following feedback loop:

```
EKF velocity noise  -->  e_v variation  -->  des RP variation
        ^                                         |
        |                                    motor can't track
        |                                    (BW ~ 2.5 Hz)
        |                                         |
        +---- position oscillation  <----  attitude delay
```

The desired roll and pitch angles are derived from the desired force vector F_des, which depends directly on measured position and velocity. Measurement noise in these quantities propagates into aggressive, high-frequency des RP commands that exceed the motor bandwidth, creating a sustained oscillation.

**Command Filtered Backstepping (CFB)** addresses this by inserting a second-order filter on the virtual control signal (desired velocity), which contains the noisy position measurement. The filter:

1. Smooths the virtual control, limiting the rate of change of des RP
2. Provides the derivative of the filtered signal analytically (no numerical differentiation)
3. Maintains Lyapunov stability through a compensation mechanism

## 2. Why Filtering p_des Does Not Work

A natural first attempt is to apply the command filter to the reference position p_des:

$$\ddot{p}_c = -2\zeta\omega_n \dot{p}_c - \omega_n^2(p_c - p_{des})$$

**Problem:** During hover, p_des is constant. After the filter transient decays, p_c -> p_des and v_c -> 0. The steady-state tracking error becomes p - p_des, identical to the unfiltered case. Since the noise originates from the measured position p and velocity v (not from p_des), filtering a constant reference provides no noise attenuation.

**The filter must be applied to a signal that contains the actual measured state.**

## 3. System Dynamics

The translational dynamics of the hexacopter, expressed as a double integrator:

$$\dot{p} = v$$

$$\dot{v} = u$$

where:
- p in R^3: position in the world frame
- v in R^3: velocity in the world frame
- u in R^3: desired acceleration (control input)

The control input u relates to the desired force by:

$$F_{des} = m(u + g \, e_3)$$

## 4. Standard Backstepping (and Its Limitation)

### Step 1 -- Position Subsystem

Define the position tracking error:

$$e_1 = p - p_{ref}$$

Choose the virtual control (desired velocity):

$$\alpha_1 = \dot{p}_{ref} - K_p \, e_1$$

If v = alpha_1, then e_1_dot = -K_p e_1, which is exponentially stable.

### Step 2 -- Velocity Subsystem

Define the velocity tracking error:

$$e_2 = v - \alpha_1$$

To design u, we need the derivative of the virtual control:

$$\dot{\alpha}_1 = \ddot{p}_{ref} - K_p \, \dot{e}_1 = \ddot{p}_{ref} - K_p(v - \dot{p}_{ref})$$

The control law:

$$u = \dot{\alpha}_1 - K_v \, e_2 - e_1$$

**Problem:** Computing alpha_1_dot requires differentiating the measured velocity v (to obtain the acceleration), or equivalently differentiating alpha_1 which contains the measured position p. This **amplifies measurement noise**, producing aggressive acceleration commands and noisy des RP.

## 5. Command Filtered Backstepping

### 5.1 Command Filter on the Virtual Control

Instead of differentiating alpha_1 analytically, pass it through a second-order command filter:

$$\ddot{\alpha}_{1,f} = -2\zeta\omega_n \, \dot{\alpha}_{1,f} - \omega_n^2 (\alpha_{1,f} - \alpha_1)$$

**Input:** alpha_1 = p_ref_dot - K_p(p - p_ref)  (contains measured position p)

**Output:**
- alpha_{1,f}: filtered desired velocity (smooth)
- alpha_1_f_dot: filtered desired acceleration (smooth, used as feedforward)

The filter acts as a second-order low-pass filter on the virtual control. Since alpha_1 contains the noisy measured position, the filter attenuates high-frequency noise while preserving the low-frequency control intent.

### 5.2 Error Definitions

**Tracking errors (w.r.t. filtered signals):**

$$e_1 = p - p_{ref}$$

$$e_2 = v - \alpha_{1,f}$$

**Filter error:**

The command filter introduces a discrepancy between alpha_1 and alpha_{1,f}. To maintain stability, define a compensation variable xi_1:

$$\dot{\xi}_1 = -K_p \, \xi_1 + (\alpha_{1,f} - \alpha_1)$$

This compensation variable tracks how the filter error propagates through the position error dynamics.

**Compensated tracking errors:**

$$\varepsilon_1 = e_1 - \xi_1$$

$$\varepsilon_2 = e_2$$

(No filter is applied at the last step, so xi_2 = 0.)

### 5.3 Control Law

$$u = \dot{\alpha}_{1,f} - K_v \, e_2 - \varepsilon_1$$

Expanding:

$$u = \dot{\alpha}_{1,f} - K_v (v - \alpha_{1,f}) - (e_1 - \xi_1)$$

where:
- alpha_1_f_dot: feedforward from command filter (smooth, no differentiation)
- K_v e_2: velocity error feedback
- varepsilon_1: compensated position error

### 5.4 Desired Force and Attitude Computation (Two-Stage)

The desired attitude is computed in two stages: first from the CFB nominal control (smooth), then adjusted with HGDO disturbance compensation to obtain the final desired attitude.

#### Stage 1: Nominal Force from CFB (Smooth)

The CFB control law produces a smooth nominal acceleration:

$$u_{cfb} = \dot{\alpha}_{1,f} - K_v \, e_2 - \varepsilon_1$$

The nominal desired force (without disturbance compensation):

$$F_{nom} = m(u_{cfb} + g \, e_3)$$

Since u_cfb is derived from the filtered alpha_{1,f} and alpha_1_f_dot, F_nom is smooth and its rate of change is limited by the filter bandwidth.

Extract the nominal collective thrust and body z-axis:

$$F_{col,nom} = \| F_{nom} \|$$

$$z_{nom} = \frac{F_{nom}}{\| F_{nom} \|}$$

Combined with the yaw reference, z_nom determines the nominal desired rotation R_nom, from which the **nominal desired roll and pitch** are obtained. These are smooth and within the motor bandwidth.

#### Stage 2: Final Desired Force with Disturbance Compensation

The HGDO estimates the disturbance force d_hgdo in the body frame. Convert to world-frame force:

$$d_{world} = R \, d_{hgdo,f}$$

The final desired force including disturbance compensation:

$$F_{des} = F_{nom} - d_{world} = m(u_{cfb} + g \, e_3) - R \, d_{hgdo,f}$$

From the final F_des, extract the **final desired attitude**:

$$F_{col} = \| F_{des} \|$$

$$z_{des} = \frac{F_{des}}{\| F_{des} \|}$$

This z_des, combined with the yaw reference, gives the final R_des and the **final desired roll and pitch** that are sent to the attitude controller.

#### Why Two Stages?

| Stage | Signal | Smooth? | Purpose |
|-------|--------|---------|---------|
| 1. F_nom | CFB output only | Yes (filtered) | Baseline attitude command within motor BW |
| 2. F_des | F_nom + HGDO comp | Less smooth | Actual attitude setpoint with disturbance rejection |

The HGDO force compensation (d_world) is intentionally **not filtered** by the command filter. This is because:

1. **HGDO bandwidth (1.6 Hz) is already comparable to the command filter bandwidth (~1.5 Hz)**, so the HGDO output is already band-limited.
2. The HGDO force contribution to des RP is relatively small (~0.5 deg std) compared to the total (~1.5 deg std in the current unfiltered system). With CFB smoothing the dominant component (position feedback), the HGDO contribution becomes an even smaller fraction.
3. Filtering the HGDO output would delay disturbance rejection, reducing its effectiveness.

If the HGDO contribution is still found to cause issues after CFB implementation, the HGDO bandwidth can be reduced by increasing eps_tau (currently 0.1, giving 1.6 Hz).

#### Dynamics Perspective

The actual dynamics with disturbance:

$$\dot{v} = u + a_d$$

With HGDO compensation applied (hat_a_d = (1/m) R d_hgdo,f):

$$\dot{v} = u_{cfb} + a_d - \hat{a}_d = u_{cfb} + \tilde{a}_d$$

where tilde_a_d = a_d - hat_a_d is the residual disturbance. If the HGDO is accurate, tilde_a_d approx 0 and the system behaves as the nominal v_dot = u_cfb, for which the CFB Lyapunov stability is guaranteed.

#### Control Architecture Flow

```
Position level (CFB):
  e_1, v  -->  alpha_1  -->  [Command Filter]  -->  alpha_{1,f}, alpha_1_f_dot
                                                          |
                                                    u_cfb (smooth)
                                                          |
                                                    F_nom = m(u_cfb + g*e3)
                                                          |  (smooth)
                                                          v
                                              F_des = F_nom - R * d_hgdo_f
                                                          |  (HGDO adjusted)
                                                          v
                                               F_col, R_des (final des attitude)
                                                          |
                                                          v
Attitude level (PD + HGDO torque):
  R, omega  -->  PD controller  +  HGDO torque comp  -->  M_des  -->  motor cmd
```

### 5.5 Tilt Constraint

After computing the final F_des, a tilt angle constraint can be applied:

$$\theta = \arccos\left(\frac{F_{des,z}}{\|F_{des}\|}\right)$$

If theta > theta_max, scale the horizontal components:

$$F_{xy,max} = F_{des,z} \cdot \tan(\theta_{max})$$

$$F_{des,x} \leftarrow F_{des,x} \cdot \frac{F_{xy,max}}{F_{xy}}, \quad F_{des,y} \leftarrow F_{des,y} \cdot \frac{F_{xy,max}}{F_{xy}}$$

This preserves the vertical force (altitude) while limiting the horizontal force (tilt angle).

## 6. Lyapunov Stability Analysis

### 6.1 Lyapunov Candidate

$$V = \frac{1}{2} \varepsilon_1^T \varepsilon_1 + \frac{1}{2} \varepsilon_2^T \varepsilon_2$$

### 6.2 Time Derivative of varepsilon_1

$$\dot{\varepsilon}_1 = \dot{e}_1 - \dot{\xi}_1$$

Compute e_1_dot:

$$\dot{e}_1 = v - \dot{p}_{ref}$$

Compute xi_1_dot:

$$\dot{\xi}_1 = -K_p \xi_1 + (\alpha_{1,f} - \alpha_1)$$

Substitute alpha_1 = p_ref_dot - K_p e_1:

$$\dot{\varepsilon}_1 = (v - \dot{p}_{ref}) - (-K_p \xi_1 + \alpha_{1,f} - \dot{p}_{ref} + K_p e_1)$$

$$= v - K_p e_1 + K_p \xi_1 - \alpha_{1,f}$$

$$= (v - \alpha_{1,f}) - K_p(e_1 - \xi_1)$$

$$\boxed{\dot{\varepsilon}_1 = \varepsilon_2 - K_p \, \varepsilon_1}$$

### 6.3 Time Derivative of varepsilon_2

$$\dot{\varepsilon}_2 = \dot{e}_2 = \dot{v} - \dot{\alpha}_{1,f} = u - \dot{\alpha}_{1,f}$$

Substitute the control law u = alpha_1_f_dot - K_v e_2 - varepsilon_1:

$$\dot{\varepsilon}_2 = (\dot{\alpha}_{1,f} - K_v e_2 - \varepsilon_1) - \dot{\alpha}_{1,f}$$

$$= -K_v e_2 - \varepsilon_1$$

Since varepsilon_2 = e_2:

$$\boxed{\dot{\varepsilon}_2 = -K_v \, \varepsilon_2 - \varepsilon_1}$$

### 6.4 Derivative of V

$$\dot{V} = \varepsilon_1^T \dot{\varepsilon}_1 + \varepsilon_2^T \dot{\varepsilon}_2$$

$$= \varepsilon_1^T (\varepsilon_2 - K_p \varepsilon_1) + \varepsilon_2^T (-K_v \varepsilon_2 - \varepsilon_1)$$

$$= \varepsilon_1^T \varepsilon_2 - K_p \varepsilon_1^T \varepsilon_1 - K_v \varepsilon_2^T \varepsilon_2 - \varepsilon_2^T \varepsilon_1$$

The cross terms cancel (varepsilon_1^T varepsilon_2 = varepsilon_2^T varepsilon_1 for scalars in each dimension):

$$\boxed{\dot{V} = -K_p \| \varepsilon_1 \|^2 - K_v \| \varepsilon_2 \|^2 \leq 0}$$

By LaSalle's invariance principle, (varepsilon_1, varepsilon_2) -> (0, 0) as t -> infinity. The closed-loop system is **asymptotically stable** for any K_p > 0, K_v > 0.

## 7. Control Architecture Summary

```
                         +------------------+
  p_ref  ------------>  |  e_1 = p - p_ref |
  p (measured) -------->  |                  |--> alpha_1 = p_ref_dot - K_p * e_1
  p_ref_dot  --------->  +------------------+
                                  |
                                  v  (noisy, contains measured p)
                      +------------------------+
                      |    Command Filter      |
                      |  2nd order, (wn, zeta) |
                      +------------------------+
                          |              |
                     alpha_{1,f}    alpha_1_f_dot
                      (smooth)       (smooth)
                          |              |
                          v              v
  v (measured) -->  e_2 = v - alpha_{1,f}
                          |
                          v
            u = alpha_1_f_dot - K_v * e_2 - varepsilon_1
                          |
                          v
                F_des = m * (u + g * e_3)
                          |
                    +-----+------+
                    |            |
                 F_col       z_des --> R_des --> des Roll, des Pitch
```

## 8. Parameter Selection

### 8.1 Filter Natural Frequency omega_n

The filter bandwidth must be **below the motor bandwidth** to ensure the motors can track the resulting attitude commands:

- Measured motor -3dB cutoff: **2.56 Hz**
- Recommended: omega_n <= 2*pi*1.5 ~ **9.4 rad/s** (1.5 Hz, providing margin)

A lower omega_n gives smoother des RP but slower position response.

### 8.2 Filter Damping Ratio zeta

- zeta = 1.0 (critically damped): no overshoot in filtered signal, conservative
- zeta = 0.707: standard Butterworth response, good trade-off
- zeta = 0.8--1.0 recommended for initial testing

### 8.3 Feedback Gains K_p, K_v

These determine the closed-loop position response in the absence of filter effects:
- K_p: position error gain (proportional)
- K_v: velocity error gain (derivative)

The natural frequency of the position loop (without filter) is sqrt(K_p), and the damping ratio is K_v / (2*sqrt(K_p)). The filter limits the effective bandwidth, so K_p and K_v can be set relatively aggressively without causing noise amplification.

## 9. Comparison: Original vs. Corrected Formulation

| Aspect | Filter on p_des (original) | Filter on alpha_1 (corrected) |
|--------|---------------------------|-------------------------------|
| Filter input | p_des (constant in hover) | alpha_1 = -K_p(p - p_ref) (contains measured p) |
| Hover steady state | p_c -> p_des, no filtering | alpha_{1,f} smooths noisy position error |
| Noise attenuation | None (filtering a constant) | Yes (filters measured position noise) |
| Des RP smoothing | No effect | Effective |
| Stability | Lyapunov guaranteed | Lyapunov guaranteed |

## 10. Implicit Trajectory Generation Without a Trajectory Planner

A key advantage of CFB is that the command filter **implicitly generates a smooth trajectory** (desired velocity and desired acceleration) even when only a constant setpoint p_ref is given, without an explicit trajectory planner.

### 10.1 The Problem Without a Trajectory

In standard PD or NMPC control for hover, the reference is:

$$p_{ref} = \text{const}, \quad \dot{p}_{ref} = 0, \quad \ddot{p}_{ref} = 0$$

There is no desired velocity profile. The controller directly computes:

$$F_{des} = m \big( -K_p(p - p_{ref}) - K_v \, v + g \, e_3 \big)$$

The entire control effort comes from feedback on measured (p, v). Any noise in these measurements directly appears in F_des and therefore in des RP.

With an explicit trajectory planner (e.g., PX4's jerk-limited generator), the planner provides smooth (p_ref(t), v_ref(t), a_ref(t)), and the feedback gains K_p, K_v can be kept small because the feedforward a_ref does most of the work. But this requires a trajectory generator.

### 10.2 How the Command Filter Generates v_des and a_des

In CFB, the virtual control alpha_1 is the "raw" desired velocity:

$$\alpha_1 = \dot{p}_{ref} - K_p(p - p_{ref}) = -K_p(p - p_{ref}) \quad \text{(hover)}$$

This signal is noisy because it contains the measured position p. The command filter smooths it:

$$\ddot{\alpha}_{1,f} = -2\zeta\omega_n \, \dot{\alpha}_{1,f} - \omega_n^2(\alpha_{1,f} - \alpha_1)$$

The filter outputs serve as implicit trajectory references:

| Filter Output | Role | Equivalent Trajectory Quantity |
|---|---|---|
| alpha_{1,f} | Smoothed desired velocity | v_des(t) |
| alpha_1_f_dot | Smoothed desired acceleration | a_des(t) (feedforward) |

### 10.3 Comparison with Explicit Trajectory Generation

```
Explicit trajectory planner:
  p_ref(t) --[jerk-limited generator]--> p_des(t), v_des(t), a_des(t)
  u = a_des + K_p(p_des - p) + K_v(v_des - v)
       ^^^^                      ^^^^
    feedforward                small feedback

CFB (no trajectory planner):
  p_ref = const
  alpha_1 = -K_p(p - p_ref)  (noisy)
  alpha_1 --[command filter]--> alpha_{1,f} = v_des(t),  alpha_1_f_dot = a_des(t)
                                               ^^^^^                      ^^^^^
                                          smooth v_des                smooth a_des
  u = a_des - K_v(v - v_des) - varepsilon_1
       ^^^^                     
    feedforward from filter    
```

Both approaches achieve the same goal: providing smooth feedforward signals so that the feedback terms are small and noise is attenuated. The difference is:

- **Explicit trajectory planner:** smooths the reference independently of the actual state
- **CFB command filter:** smooths a signal derived from the actual state (position error)

For hover, the CFB approach is more appropriate because:
1. No trajectory planner is needed
2. The filter directly acts on the noisy quantity (position error)
3. The filter bandwidth limits how fast des RP can change, matching motor capability

### 10.4 Intuitive Interpretation

The command filter converts a **static setpoint** into a **dynamic reference trajectory**:

- When the drone is displaced from p_ref, alpha_1 = -K_p * e_1 commands a velocity toward the setpoint
- The filter smooths this into a gradual, bandwidth-limited velocity profile alpha_{1,f}
- The filter derivative alpha_1_f_dot provides the corresponding smooth acceleration
- The controller uses alpha_1_f_dot as feedforward, with only small corrections from velocity feedback

Even without an explicit trajectory generator, the CFB controller behaves as if it has one. The filter bandwidth omega_n determines how aggressively the "implicit trajectory" drives the drone back to the setpoint:

- High omega_n: fast return, but des RP changes quickly (may exceed motor BW)
- Low omega_n: slow return, but des RP stays smooth (within motor BW)

Setting omega_n below the motor bandwidth (~2.5 Hz) ensures that the implicit trajectory is always physically realizable.

### 10.5 Hover Steady State

At hover equilibrium (p = p_ref, v = 0):

$$\alpha_1 = 0, \quad \alpha_{1,f} = 0, \quad \dot{\alpha}_{1,f} = 0$$

$$u = 0 - K_v(0) - 0 = 0$$

$$F_{des} = m \, g \, e_3$$

The controller outputs pure hover thrust with zero horizontal force, and des RP = 0. Any deviation from equilibrium produces a smooth, filtered response through the command filter.

## 11. Implementation Notes

1. **State variables to maintain:**
   - alpha_{1,f} in R^3 (filtered virtual control)
   - alpha_1_f_dot in R^3 (derivative of filtered virtual control)
   - xi_1 in R^3 (compensation variable)
   - Total: 9 additional states

2. **Initialization:**
   - alpha_{1,f}(0) = alpha_1(0) = p_ref_dot(0) - K_p(p(0) - p_ref(0))
   - alpha_1_f_dot(0) = 0
   - xi_1(0) = 0

3. **Integration:** The command filter and compensation dynamics are integrated at the control loop rate using Euler or RK4.

4. **Saturation:** If alpha_{1,f} or alpha_1_f_dot become excessively large (e.g., due to large initial error), apply saturation to prevent wind-up.

5. **Existing NMPC integration:** The CFB output u (desired acceleration) replaces the NMPC force output. The HGDO disturbance compensation can be added to u:
   
   u_total = u_cfb + (1/m) * d_hgdo
   
   where d_hgdo is the HGDO estimated disturbance force.
