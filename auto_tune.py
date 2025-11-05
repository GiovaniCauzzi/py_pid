import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pid_lib import PIDController

HEATER_MAX_POWER = 50.0  # W

# --- Plant parameters ---
C1, C2 = 5.0, 20.0  # J/°C
R12, R1a, R2a = 2.0, 5.0, 10.0  # °C/W
Ta = 20.0  # ambient °C

# --- Create PID controller ---
pid = PIDController(
    Kp=3.0, Ki=0.2, Kd=0.0,
    setpoint=60.0,
    output_limits=(0, HEATER_MAX_POWER),          # Power limits (0–10 W)
    integral_limits=(0, HEATER_MAX_POWER)       # Optional anti-windup
)

# --- Simulation setup ---
dt = 0.1
t_final = 500
n_steps = int(t_final / dt)
T1, T2 = Ta, Ta  # initial temps

# --- Logging ---
t_log, T1_log, T2_log, P_log, setpoint_log = [], [], [], [], []


# auto-tuning stage



for i in range(n_steps):
    t = i * dt

    if t >= t_final/2:
        pid.setpoint = 40.0  # change setpoint mid-simulation

    # PID update using sensor measurement
    P = pid.update(T2, dt)
    
    # Plant model
    def thermal_model(t, T):
        T1, T2 = T
        dT1dt = (P - (T1 - T2)/R12 - (T1 - Ta)/R1a) / C1
        dT2dt = ((T1 - T2)/R12 - (T2 - Ta)/R2a) / C2
        return [dT1dt, dT2dt]
    
    # Integrate over the time step
    sol = solve_ivp(thermal_model, [t, t + dt], [T1, T2], t_eval=[t + dt])
    T1, T2 = sol.y[:, -1]
    
    # Log data
    t_log.append(t)
    T1_log.append(T1)
    T2_log.append(T2)
    P_log.append(P)
    setpoint_log.append(pid.setpoint)

# --- Plot ---
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(t_log, T1_log, label='Heater T₁')
plt.plot(t_log, T2_log, label='Sensor T₂ (feedback)')
plt.plot(t_log, setpoint_log, label='Sensor T₂ (feedback)')
# plt.axhline(pid.setpoint, color='r', linestyle='--', label='Setpoint')
plt.ylabel("Temperature [°C]")
plt.legend()
plt.grid(True)

plt.subplot(2,1,2)
plt.plot(t_log, P_log, label='Power (PID output)')
plt.ylabel("Power [W]")
plt.xlabel("Time [s]")
plt.grid(True)
plt.legend()

plt.show()
