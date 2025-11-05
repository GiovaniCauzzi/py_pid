import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === Plant parameters ===
C1, C2 = 5.0, 20.0  # J/°C
R12, R1a, R2a = 2.0, 5.0, 10.0  # °C/W
Ta = 30.0  # ambient °C

# === PID parameters ===
Kp = 1 #3.0
Ki = 0.01 #0.2
Kd = 1 #1.0
setpoint = 50.0  # °C

# === Simulation setup ===
dt = 0.1  # step size (s)
t_final = 500
n_steps = int(t_final / dt)

# === Initial conditions ===
T1 = Ta
T2 = Ta
integral = 0.0
prev_error = 0.0

# === Logging ===
t_log, T1_log, T2_log, P_log = [], [], [], []

# === Simulation loop ===
for i in range(n_steps):
    t = i * dt
    error = setpoint - T2

    # --- PID computation ---
    integral += error * dt
    derivative = (error - prev_error) / dt
    P = Kp * error + Ki * integral + Kd * derivative

    # Limit power (e.g. heater max 0–10 W)
    P = np.clip(P, 0, 10)

    # --- Integrate the plant over the next dt ---
    def thermal_model(t, T):
        T1, T2 = T
        dT1dt = (P - (T1 - T2)/R12 - (T1 - Ta)/R1a) / C1
        dT2dt = ((T1 - T2)/R12 - (T2 - Ta)/R2a) / C2
        return [dT1dt, dT2dt]

    sol = solve_ivp(thermal_model, [t, t + dt], [T1, T2], t_eval=[t + dt])
    T1, T2 = sol.y[:, -1]

    # --- Log data ---
    t_log.append(t)
    T1_log.append(T1)
    T2_log.append(T2)
    P_log.append(P)

    prev_error = error

# === Plot results ===
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)
plt.plot(t_log, T1_log, label='Heater T₁')
plt.plot(t_log, T2_log, label='Sensor T₂ (feedback)')
plt.axhline(setpoint, color='r', linestyle='--', label='Setpoint')
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
