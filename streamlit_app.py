import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Title
st.title("Interactive 2D Kalman Filter Demo")

# Simulation parameters
timesteps = st.slider("Select Number of Time Steps", 50, 200, 100)
attack_start = st.slider("Start of Cyber Attack", 10, 80, 20)
attack_end = st.slider("End of Cyber Attack", attack_start + 1, timesteps, 60)

# Generate data
x_true = np.zeros((timesteps, 2))
A = np.array([[1, 1], [0, 1]])
Q = np.array([[0.1, 0], [0, 0.1]])
for t in range(1, timesteps):
    x_true[t] = A @ x_true[t-1] + np.random.multivariate_normal([0, 0], Q)

# Add noisy measurements
R = np.array([[0.5]])
z = x_true[:, 0] + np.random.normal(0, R[0][0], timesteps)
z[attack_start:attack_end] += np.random.normal(0, 3, attack_end - attack_start)

# Anomaly detection
iso_forest = IsolationForest(contamination=0.05)
anomalies = iso_forest.fit_predict(z.reshape(-1, 1))

# Plot
fig, ax = plt.subplots()
ax.plot(x_true[:, 0], label='True Position', linestyle='dashed')
ax.plot(z, label='Measured (with attacks)', alpha=0.5)
ax.axvspan(attack_start, attack_end, color='red', alpha=0.2, label='Attack Period')
ax.legend()
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.title("Kalman Filter Interactive Simulation")

# Display in Streamlit
st.pyplot(fig)
