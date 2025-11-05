class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0.0, output_limits=(None, None), integral_limits=(None, None)):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None
        
        self.output_limits = output_limits
        self.integral_limits = integral_limits

    def reset(self):
        """Reset the internal state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_measurement = None

    def update(self, measurement, dt):
        """
        Compute PID output based on measurement and time step.
        """
        error = self.setpoint - measurement
        
        # --- Integral term ---
        self.integral += error * dt
        # Clamp integral (anti-windup)
        if self.integral_limits[0] is not None:
            self.integral = max(self.integral, self.integral_limits[0])
        if self.integral_limits[1] is not None:
            self.integral = min(self.integral, self.integral_limits[1])
        
        # --- Derivative term ---
        derivative = (error - self.prev_error) / dt if self.prev_measurement is not None else 0.0
        
        # --- PID Output ---
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        # --- Output clamping ---
        if self.output_limits[0] is not None:
            output = max(output, self.output_limits[0])
        if self.output_limits[1] is not None:
            output = min(output, self.output_limits[1])
        
        # --- Save state ---
        self.prev_error = error
        self.prev_measurement = measurement
        
        return output
