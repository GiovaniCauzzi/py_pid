# py_pid

A simple PID controller implementation, simulated against a two-node thermal plant model (heater + sensor, coupled through thermal resistances) using `scipy.integrate.solve_ivp`.

## Project structure

- [pid_lib.py](pid_lib.py) — `PIDController` class (output and integral anti-windup clamping included).
- [pid_test.py](pid_test.py) — runs a simulation with fixed PID gains and plots the response.
- [auto_tune.py](auto_tune.py) — variant of the simulation intended for PID auto-tuning experiments.

## Setup

### 1. Create the virtual environment

```powershell
python -m venv .venv
```

### 2. Activate it

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt should show `(.venv)` once active. If PowerShell blocks the script, run once (as your user, not admin):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Deactivate when done

```powershell
deactivate
```

## Usage

With the venv activated:

```powershell
python pid_test.py
```

```powershell
python auto_tune.py
```

Each script runs a simulation and opens a matplotlib plot of temperature and controller output over time.
