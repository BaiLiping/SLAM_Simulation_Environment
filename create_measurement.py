import numpy as np

def create_measurement(x, steps, v, ang_v, T, R, L_BS, L_VA, L_SP):
    """
    Generates measurements for a user moving in a 3D space with reflections from Virtual Anchors (VA) and Scatter Points (SP).

    Parameters:
    - x (np.ndarray): Initial state vector of the user [x, y, z, heading, bias].
    - steps (int): Number of time steps to simulate.
    - v (float): User speed (m/s).
    - ang_v (float): User angular velocity (rad/s).
    - T (float): Sampling time (seconds).
    - R (np.ndarray): Measurement noise covariance matrix.
    - L_BS (np.ndarray): Positions of Base Stations (BS), shape (3, N_BS).
    - L_VA (np.ndarray): Positions of Virtual Anchors (VA), shape (N_VA, 3).
    - L_SP (np.ndarray): Positions of Scatter Points (SP), shape (3, N_SP).

    Returns:
    - measurement (list): List of measurements for each time step.
    - v_state (np.ndarray): Real user state history over time.
    - index (list): List of measurement source indices for each time step.
    - pos_s_all (list): List of pos_s (reflection points) for each time step.
    - pos_vb_all (list): List of VB points for each SP at each time step.
    - association_record (list): Association records for each time step.
    - clutter_positions_all (list): Clutter positions for each time step.
    """
    # Ensure x has at least 5 elements [x, y, z, heading, bias]
    if x.shape[0] < 5:
        x = np.hstack((x[0:2], [0], x[2:]))  # Insert z=0 if missing

    measurement = [None] * steps          # Initialize measurement list
    index = [None] * steps                # Initialize index list
    pos_s_all = []                        # List to hold pos_s_list for all time steps (for plotting purposes)
    pos_vb_all = []                       # List to hold VB points for all time steps
    v_state = []                          # Initialize state history (tracks user state at each step)
    association_record = []               # Initialize the association record for plotting purposes
    clutter_positions_all = []            # Initialize clutter positions for plotting

    P_detection = 1  # Detection probability
    Fov = 100           # Field of view (maximum range for detecting scatter points)

    BS0 = L_BS[:, 0]    # Assuming the first BS is the reference BS

    for i in range(steps):
        v_a = v / ang_v  # Radius of the arc the user is moving along
        ang_delta = ang_v * T  # Change in angle over time step

        # Update user state using a circular motion model
        x[0] = x[0] + v_a * (np.sin(x[3] + ang_delta) - np.sin(x[3]))
        x[1] = x[1] + v_a * (-np.cos(x[3] + ang_delta) + np.cos(x[3]))
        x[3] = x[3] + ang_delta

        # Process Noise
        std_devs = np.array([0.2, 0.2, 0, 0.2 / 180 * np.pi, 0.2])
        noise = std_devs * np.random.randn(5)
        x = x + noise  # Add process noise to the trajectory

        v_state.append(x.copy())  # Record the updated state

        z1 = []          # Initialize the measurement list for this time step
        index_s = []     # Initialize the index list (stores which object is detected)
        pos_s_list = []  # Initialize the pos_s list for this time step
        pos_vb_list = [] # Initialize the pos_vb list for VB points
        step_assoc = []  # Initialize association record for this step
        clutter_positions = []  # Initialize clutter positions for this step

        # Extract current user position from state x
        user_position = x[0:3]

        # Loop through each Base Station (BS)
        for j in range(L_BS.shape[1]):
            if np.random.rand() < P_detection:  # Simulate detection probability
                z = np.zeros(5)  # Measurement vector for this BS
                # Compute range and angle measurements
                range_bs = np.linalg.norm(L_BS[:, j] - user_position)  # Range between the base station and user
                z[0] = range_bs
                z[1] = np.arctan2(user_position[1], user_position[0])               # AOD Azimuth angle from the transmitter
                z[2] = np.arcsin((user_position[2] - L_BS[2, j]) / range_bs)        # AoD Elevation from the transmitter
                z[3] = np.pi + np.arctan2(user_position[1], user_position[0]) - x[3]  # AoA Azimuth from the receiver
                z[4] = np.arcsin((L_BS[2, j] - user_position[2]) / range_bs)        # AoA Elevation from the receiver

                # Store the measurement and index
                noise_measure = np.random.multivariate_normal(mean=np.zeros(5), cov=R)
                z_noisy = z + noise_measure
                z1.append(z_noisy)
                index_s.append(1)          # Index 1 corresponds to BS
                pos_s_list.append(None)    # No pos_s for BS
                step_assoc.append(('BS', j))  # Record association

        # Loop through each Virtual Anchor (VA)
        for j in range(L_VA.shape[0]):
            if (i + 1) in [1, 2, 3, 4, 5] or np.random.rand() < P_detection:
                z = np.zeros(5)  # Measurement vector for this VA

                # Position calculation for VA
                u = (L_BS[:, 0] - L_VA[j, :]) / np.linalg.norm(L_BS[:, 0] - L_VA[j, :])
                f = (L_BS[:, 0] + L_VA[j, :]) / 2
                numerator = np.dot((f - L_VA[j, :]), u)
                denominator = np.dot((user_position - L_VA[j, :]), u)

                # Check for zero denominator
                if np.abs(denominator) < 1e-6:
                    pos_s = None
                else:
                    pos_s = L_VA[j, :] + (numerator / denominator) * (user_position - L_VA[j, :])

                # Compute range and angle measurements
                z[0] = np.linalg.norm(L_VA[j, :] - user_position)
                if pos_s is not None:
                    z[1] = np.arctan2(pos_s[1], pos_s[0])
                    z[2] = np.arcsin((pos_s[2] - L_BS[2, 0]) / np.linalg.norm(pos_s - L_BS[:, 0]))
                    z[3] = np.arctan2(L_VA[j,1] - user_position[1], L_VA[j,0] - user_position[0]) - x[3]
                    z[4] = np.arcsin((L_VA[j,2] - user_position[2]) / np.linalg.norm(user_position - L_VA[j, :]))
                else:
                    # Handle cases where pos_s is None
                    z[1] = 0
                    z[2] = 0
                    z[3] = 0
                    z[4] = 0

                # Store the measurement, index, and pos_s
                noise_measure = np.random.multivariate_normal(mean=np.zeros(5), cov=R)
                z_noisy = z + noise_measure
                z1.append(z_noisy)
                index_s.append(1 + j)        # Index for VA
                pos_s_list.append(pos_s)     # Store pos_s for VA
                step_assoc.append(('VA', j)) # Record association

        # Loop through each Scatter Point (SP)
        for j in range(L_SP.shape[1]):
            # Detect scatter points at specific steps or with a random chance
            if (i + 1) in [1, 2, 3, 7, 8, 9, 10, 17, 18, 19, 27, 28, 29, 30, 32, 33, 36, 37, 38, 39] or np.random.rand() < P_detection:
                range_sp_user = np.linalg.norm(L_SP[j] - user_position)
                range_bs_sp = np.linalg.norm(L_SP[j] - BS0)
                if range_sp_user <= Fov:  # Only detect points within the field of view
                    z = np.zeros(5)  # Measurement vector for this SP
                    # Compute range and angle measurements
                    z[0] = range_sp_user + range_bs_sp  # Total Range
                    z[1] = np.arctan2(L_SP[j, 1], L_SP[j, 0])  # AoD Azimuth
                    z[2] = np.arcsin((L_SP[j, 2] - BS0[2]) / range_bs_sp)  # AoD Elevation
                    z[3] = np.arctan2(L_SP[j, 1] - user_position[1], L_SP[0, j] - user_position[0]) - x[3]  # Bearing angle
                    z[4] = np.arcsin((L_SP[j, 2] - user_position[2]) / range_sp_user)  # Elevation AoA

                    # Add noise to the measurement and store it
                    noise_measure = np.random.multivariate_normal(mean=np.zeros(5), cov=R)
                    z_noisy = z + noise_measure
                    z1.append(z_noisy)
                    index_s.append(5 + j)      # Index for SP
                    pos_s_list.append(L_SP[0:3, j].copy())    # Store SP position
                    step_assoc.append(('SP', j))  # Record association

        # Generate clutter (random false measurements)
        number = np.random.poisson(5)  # Number of clutter points (Poisson distributed)
        for j in range(number):
            clutter = np.array([
                200 * np.random.rand() - 100,
                200 * np.random.rand() - 100,
                40 * np.random.rand()
            ])  # Random clutter position
            # Compute measurements for clutter points
            range_clutter = np.linalg.norm(clutter[0:3] - user_position)
            range_bs_clutter = np.linalg.norm(clutter[0:3] - BS0)
            z_clutter = np.zeros(5)  # Measurement vector for clutter
            z_clutter[0] = range_clutter + range_bs_clutter  # Total Range
            z_clutter[1] = -np.arctan2(
                clutter[1] - BS0[1],
                clutter[0] - BS0[0]
            )  # AoD Azimuth
            z_clutter[2] = np.arcsin(
                (clutter[2] - BS0[2]) / range_bs_clutter
            )  # AoD Elevation
            z_clutter[3] = np.arctan2(
                clutter[1] - user_position[1],
                clutter[0] - user_position[0]
            ) - x[3]  # Bearing angle
            z_clutter[4] = np.arcsin(
                (clutter[2] - user_position[2]) /
                np.linalg.norm(user_position - clutter[0:3])
            )  # Elevation AoA

            # Add noise to the clutter measurement and store it
            noise_measure = np.random.multivariate_normal(mean=np.zeros(5), cov=R)
            z_clutter_noisy = z_clutter + noise_measure
            z1.append(z_clutter_noisy)
            index_s.append(0)           # Index 0 represents clutter
            pos_s_list.append(None)     # Append None for clutter
            clutter_positions.append(clutter)  # Store clutter position

            step_assoc.append(('clutter', j))  # Record association

        # Store the measurements and indices for this time step
        measurement[i] = z1
        index[i] = index_s
        pos_s_all.append(pos_s_list)               # Store pos_s_list for this time step
        pos_vb_all.append(pos_vb_list)             # Store pos_vb_list for VB points
        association_record.append(step_assoc)        # Store association record for this step
        clutter_positions_all.append(clutter_positions)  # Store clutter positions for this step

    # Remove the third dimension from v_state (convert it back to 2D for further processing)
    v_state = np.array(v_state).T
    if v_state.shape[0] == 5:
        # Remove bias or adjust as necessary
        v_state = np.delete(v_state, 4, axis=0)  # Assuming the 5th element is bias
    elif v_state.shape[0] > 3:
        v_state = v_state[:3, :]  # Keep only x, y, z

    return measurement, v_state, index, pos_s_all, association_record, clutter_positions_all
