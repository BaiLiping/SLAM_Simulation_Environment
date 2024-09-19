import numpy as np

def create_measurement(x, steps, v, ang_v, T, R, L_BS, L_VA, L_SP):
    x = np.hstack((x[0:2], [0], x[2:]))

    measurement = [None] * steps  # Initialize measurement list
    index = [None] * steps        # Initialize index list
    pos_s_all = []                # List to hold pos_s_list for all time steps this is for plotting purpose only

    P_detection = 1  # Detection probability
    Fov = 50  # Field of view (maximum range for detecting scatter points)

    v_state = []  # Initialize state history (tracks user state at each step)

    for i in range(steps):
        v_a = v / ang_v  # Radius of the arc the user is moving along
        ang_delta = ang_v * T  # Change in angle over time step

        # Update user state using a circular motion model
        x[0] = x[0] + v_a * (np.sin(x[3] + ang_delta) - np.sin(x[3]))
        x[1] = x[1] + v_a * (-np.cos(x[3] + ang_delta) + np.cos(x[3]))
        x[3] = x[3] + ang_delta
        # Process Noise
        std_devs = np.array([0.2, 0.2, 0, 0.2 / 180 * np.pi, 0.2])
        # Generate random noise
        noise = std_devs * np.random.randn(5)
        # Add process noise to the trajectory
        x = x + noise
        # v_state updated
        v_state.append(x.copy())  # Record the updated state

        z1 = []          # Initialize the measurement list for this time step
        index_s = []     # Initialize the index list (stores which object is detected)
        pos_s_list = []  # Initialize the pos_s list for this time step

        # Loop through each Base Station (BS)
        for j in range(L_BS.shape[1]):
            if np.random.rand() < P_detection:  # Simulate detection probability
                z = np.zeros(5)  # Measurement vector for this BS
                # Compute range and angle measurements
                z[0] = np.linalg.norm(L_BS[:, j] - x[0:3]) # Range between the base station and user
                z[1] = np.arctan2(x[1], x[0])  # AOD Azimuth angle from the transmitter
                z[2] = np.arcsin((x[2] - L_BS[2, j]) / np.linalg.norm(x[0:3] - L_BS[:, j])) # AoD Elevation from the transmitter
                z[3] = np.pi + np.arctan2(x[1], x[0]) - x[3]  # AoA Azimuth from the receiver
                z[4] = np.arcsin((L_BS[2, j] - x[2]) / np.linalg.norm(x[0:3] - L_BS[:, j])) # AoA Elevation from the receiver

                # Store the measurement and index
                noise = np.random.multivariate_normal(mean=np.zeros(5), cov=R)
                z_noisy = (z + noise)
                z1.append(z_noisy)
                index_s.append(1)  # Index 1 corresponds to BS
                pos_s_list.append(None)  # No pos_s for BS

        # Loop through each Virtual Anchor (VA)
        for j in range(L_VA.shape[0]):
            if (i + 1) in [1, 2, 3, 4, 5] or np.random.rand() < P_detection:
                z = np.zeros(5)  # Measurement vector for this VA

                # Position calculation for VA
                u = (L_BS[:, 0] - L_VA[j, :]) / np.linalg.norm(L_BS[:, 0] - L_VA[j, :])
                f = (L_BS[:, 0] + L_VA[j, :]) / 2
                numerator = np.dot((f - L_VA[j, :]), u)
                denominator = np.dot((x[0:3] - L_VA[j, :]), u)

                # Check for zero denominator
                if np.abs(denominator) < 1e-6:
                    pos_s = None
                else:
                    pos_s = L_VA[j, :] + (numerator / denominator) * (x[0:3] - L_VA[j, :])

                # Compute range and angle measurements
                z[0] = np.linalg.norm(L_VA[j, :] - x[0:3])
                if pos_s is not None:
                    z[1] = -np.arctan2(pos_s[1], pos_s[0])
                    z[2] = np.arcsin((pos_s[2] - L_BS[2, 0]) / np.linalg.norm(pos_s - L_BS[:, 0]))
                    z[3] = np.arctan2(L_VA[j,1] - x[1], L_VA[j,0] - x[0]) - x[3]
                    z[4] = np.arcsin((L_VA[j,2] - x[2]) / np.linalg.norm(x[0:3] - L_VA[j]))
                else:
                    # Handle cases where pos_s is None
                    z[1] = 0
                    z[2] = 0
                    z[3] = 0
                    z[4] = 0

                # Store the measurement, index, and pos_s
                noise = np.random.multivariate_normal(mean=np.zeros(5), cov=R)
                z_noisy = (z + noise)
                z1.append(z_noisy)
                index_s.append(1 + j)  # Index for VA
                pos_s_list.append(pos_s)  # Store pos_s for VA


        # Store the measurements and indices for this time step
        measurement[i] = z1
        index[i] = index_s
        pos_s_all.append(pos_s_list)  # Store pos_s_list for this time step

    # Remove the third dimension from v_state (convert it back to 2D for further processing)
    v_state = np.array(v_state).T
    v_state = np.delete(v_state, 2, axis=0)

    return measurement, v_state, index, pos_s_all
