import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Simulation Parameters
timesteps = 100  # Total time steps
attack_start, attack_end = 20, 60  # Attack window

# Initialize State
x_true = np.zeros((timesteps, 2))  # True positions
x_est = np.zeros((timesteps, 2))  # Estimated positions

# Kalman Filter Matrices
A = np.array([[1, 1], [0, 1]])  # State transition matrix
H = np.array([[1, 0]])  # Measurement matrix
Q = np.array([[0.1, 0], [0, 0.1]])  # Process noise covariance
R = np.array([[0.5]])  # Measurement noise covariance
P = np.eye(2)  # Covariance matrix
x = np.array([[0], [1]])  # Initial state

# Generate True Position Data with Noise
for t in range(1, timesteps):
    x_true[t] = A @ x_true[t-1] + np.random.multivariate_normal([0, 0], Q)

# Introduce Cyber Attacks (Denial of Service & Data Injection)
z = x_true[:, 0] + np.random.normal(0, R[0][0], timesteps)  # Noisy measurements
z[attack_start:attack_end] += np.random.normal(0, 3, attack_end - attack_start)  # Injected noise

# Initialize Isolation Forest for Anomaly Detection
iso_forest = IsolationForest(contamination=0.05)
anomalies = iso_forest.fit_predict(z.reshape(-1, 1))

# Kalman Filter Estimation
for t in range(1, timesteps):
    # Prediction Step
    x_pred = A @ x
    P_pred = A @ P @ A.T + Q
    
    # Update Step
    if anomalies[t] == 1:  # Normal data
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        x = x_pred + K @ (z[t] - H @ x_pred)
        P = (np.eye(2) - K @ H) @ P_pred
    else:  # Anomalous data (Adaptive Kalman Filter Adjustment)
        R = np.array([[2.0]])  # Increase measurement noise assumption
        K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
        x = x_pred + K @ (z[t] - H @ x_pred)
        P = (np.eye(2) - K @ H) @ P_pred
    
    x_est[t] = x.ravel()

# Plot Results
plt.figure(figsize=(10, 5))
plt.plot(x_true[:, 0], label='True Position', linestyle='dashed')
plt.plot(z, label='Measured (with attacks)', alpha=0.5)
plt.plot(x_est[:, 0], label='Kalman Estimate')
plt.axvspan(attack_start, attack_end, color='red', alpha=0.2, label='Attack Period')
plt.legend()
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.title("2D Kalman Filter Demo with Cyber Attack Simulation")
plt.show()
