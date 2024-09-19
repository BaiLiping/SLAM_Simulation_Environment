import numpy as np
from create_measurement import create_measurement
import plotly.graph_objects as go
import plotly.io as pio
import imageio.v2 as imageio             # For creating GIFs without deprecation warnings
import os                                # For file management

# Monte Carlo Counter
MCC = 0

# Number of Monte Carlo simulations
num_MC = 1  # Set to 1 if you only want one GIF per simulation

# Directory to save images and interactive HTML files
image_dir = 'gif_frames'
if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Name of the output GIF file
gif_filename = 'MonteCarlo.gif'

# List to hold image filenames
images = []

# Run Monte Carlo simulations
for mcc in range(1, num_MC + 1):

    ###############################################################
    # Initialization of Simulation Parameters and Variables
    ###############################################################

    # Initialize Base Station (BS) and Virtual Anchors (VA)
    BS = np.array([[0], [0], [40]])  # Base Station coordinates (x, y, z)
    VA = np.array([
        [200, 0, 40],
        [-200, 0, 40],
        [0, 200, 40],
        [0, -200, 40]
    ])  # Each row is a VA

    # Initialize Scatter Points (SP) with fixed values
    SP = np.array([
        [99, 0, 10],
        [-99, 0, 10],
        [0, 99, 10],
        [0, -99, 10]
    ])  # Each row is a SP

    # Number of particles for the particle filter
    N = 10
    # Number of time steps
    K = 41
    # Process noise covariance matrix (for user movement)
    Q = (np.diag([0.2, 0.2, 0.2 * np.pi / 180, 0.2])) ** 2
    # Initial particle generation covariance
    Q0 = (np.diag([0.3, 0.3, 0.3 * np.pi / 180, 0.3])) ** 2
    # Measurement noise covariance matrix
    R = np.diag([0.1, 0.01, 0.01, 0.01, 0.01]) ** 2
    # Sampling time
    T = 0.5  # seconds
    # Number of best hypotheses for PMBM
    k_best = 10

    # Initial state of the user [x, y, heading, bias]
    s1_in = np.array([70.7285, 0, np.pi / 2, 300])  # Initial position and state
    s1_v = 22.22       # User speed (m/s)
    s1_ang_v = np.pi / 10  # User angular velocity (rad/s)

    ###############################################################
    # Measurement Generation
    ###############################################################

    # Generate measurements for this Monte Carlo simulation
    measurement1, v1real_state, index, pos_s_all = create_measurement(s1_in, K, s1_v, s1_ang_v, T, R, BS, VA, SP)

    ###############################################################
    # Visualization (Interactive 3D Plotting with Plotly)
    ###############################################################

    # Loop over all time steps and plot the positions incrementally
    for step in range(K):
        # Create a new figure for each iteration
        fig = go.Figure()

        # Plot Base Station (BS)
        fig.add_trace(go.Scatter3d(
            x=[BS[0, 0]],
            y=[BS[1, 0]],
            z=[BS[2, 0]],
            mode='markers+text',
            marker=dict(size=5, color='red'),
            text=['BS'],
            textposition='top center',
            name='BS'
        ))

        # Plot Virtual Anchors (VA)
        fig.add_trace(go.Scatter3d(
            x=VA[:, 0],
            y=VA[:, 1],
            z=VA[:, 2],
            mode='markers+text',
            marker=dict(size=5, color='blue'),
            text=[f'VA{va_idx+1}' for va_idx in range(VA.shape[0])],
            textposition='top center',
            name='VA'
        ))

        # Plot Scatter Points (SP)
        fig.add_trace(go.Scatter3d(
            x=SP[:, 0],
            y=SP[:, 1],
            z=SP[:, 2],
            mode='markers+text',
            marker=dict(size=5, color='green'),
            text=[f'SP{sp_idx+1}' for sp_idx in range(SP.shape[0])],
            textposition='top center',
            name='SP'
        ))

        # Plot user's trajectory up to current step
        fig.add_trace(go.Scatter3d(
            x=v1real_state[0, :step+1],
            y=v1real_state[1, :step+1],
            z=np.zeros(step+1),
            mode='lines+markers',
            line=dict(color='black', width=2),
            marker=dict(size=3, color='black'),
            name='User Trajectory'
        ))

        # Plot user's current position
        user_position = np.array([v1real_state[0, step], v1real_state[1, step], 0])
        fig.add_trace(go.Scatter3d(
            x=[user_position[0]],
            y=[user_position[1]],
            z=[user_position[2]],
            mode='markers+text',
            marker=dict(size=5, color='black'),
            text=['User'],
            textposition='top center',
            name='User Position'
        ))

        # Plot an arrow indicating the user's heading
        user_heading = v1real_state[2, step]  # User's heading at current step
        arrow_length = 50  # Adjust the length as necessary
        heading_vector = np.array([np.cos(user_heading), np.sin(user_heading), 0]) * arrow_length
        fig.add_trace(go.Cone(
            x=[user_position[0]],
            y=[user_position[1]],
            z=[user_position[2]],
            u=[heading_vector[0]],
            v=[heading_vector[1]],
            w=[heading_vector[2]],
            colorscale=[[0, 'black'], [1, 'black']],
            showscale=False,
            sizemode='absolute',
            sizeref=10,
            anchor="tail",
            name='User Heading'
        ))

        # Plot measurements at current step
        z_k = measurement1[step]       # Measurements at current step (list of arrays)
        index_s = index[step]          # Indices of measurement sources (list)
        pos_s_list = pos_s_all[step]   # pos_s for measurements at this time step

        # For each measurement, plot the AoA and reflection paths
        for j in range(len(z_k)):
            # Extract measurement data
            z = z_k[j]
            range_ = z[0]          # Range (TOA)
            azimuth_AoA = z[3] + user_heading   # Adjust AoA azimuth with user's heading
            elevation_AoA = z[4]   # Elevation AoA
            idx = index_s[j]       # Measurement source index
            pos_s = pos_s_list[j]  # pos_s for this measurement (None if not VA)

            # For VA reflections, set the AoA line to pos_s
            if idx >= 1 and idx <= (1 + VA.shape[0]) and pos_s is not None:
                AoA_end = pos_s  # The AoA line ends at pos_s
                # Plot reflection path from BS to pos_s
                fig.add_trace(go.Scatter3d(
                    x=[BS[0, 0], pos_s[0]],
                    y=[BS[1, 0], pos_s[1]],
                    z=[BS[2, 0], pos_s[2]],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    showlegend=False,
                    name='Reflection Path'
                ))
                # Plot reflection path from pos_s to user
                fig.add_trace(go.Scatter3d(
                    x=[pos_s[0], user_position[0]],
                    y=[pos_s[1], user_position[1]],
                    z=[pos_s[2], user_position[2]],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=2),
                    showlegend=False,
                    name='Reflection Path'
                ))
                # Mark pos_s
                fig.add_trace(go.Scatter3d(
                    x=[pos_s[0]],
                    y=[pos_s[1]],
                    z=[pos_s[2]],
                    mode='markers+text',
                    marker=dict(size=1, color='magenta', symbol='x'),
                    #text=['pos_s'],
                    #textposition='top center',
                    name='Reflection Point'
                ))
                # AoA line from user to pos_s
                #fig.add_trace(go.Scatter3d(
                #    x=[user_position[0], pos_s[0]],
                #    y=[user_position[1], pos_s[1]],
                #    z=[user_position[2], pos_s[2]],
                #    mode='lines',
                #    line=dict(color='red', dash='dash', width=2),
                #    showlegend=False,
                #    name='AoA Line'
                #))
                # For other cases, compute AoA line based on azimuth and elevation
                u_AoA = np.array([
                    np.cos(elevation_AoA) * np.cos(azimuth_AoA),
                    np.cos(elevation_AoA) * np.sin(azimuth_AoA),
                    np.sin(elevation_AoA)
                ])
                AoA_end = user_position + u_AoA * range_
                # AoA line from user
                fig.add_trace(go.Scatter3d(
                    x=[user_position[0], AoA_end[0]],
                    y=[user_position[1], AoA_end[1]],
                    z=[user_position[2], AoA_end[2]],
                    mode='lines',
                    line=dict(color='black', dash='dash', width=3),
                    showlegend=False,
                    name='AoA Line'
                ))
                # Mark AoA end point
                fig.add_trace(go.Scatter3d(
                    x=[AoA_end[0]],
                    y=[AoA_end[1]],
                    z=[AoA_end[2]],
                    mode='markers',
                    marker=dict(size=3, color='black'),
                    showlegend=False,
                    name='AoA End'
                ))

        # Update layout
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=[-250, 250]),
                yaxis=dict(range=[-250, 250]),
                zaxis=dict(range=[0, 50]),
                aspectratio=dict(x=1, y=1, z=0.5),
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            title=f'Step {step + 1}',
            showlegend=True
        )

        # Save the figure as an HTML file
        html_filename = os.path.join(image_dir, f'frame_{step:03d}.html')
        pio.write_html(fig, file=html_filename, auto_open=False)

        # Save the figure as a PNG image (optional, for GIF creation)
        png_filename = os.path.join(image_dir, f'frame_{step:03d}.png')
        fig.write_image(png_filename)
        images.append(png_filename)

    # Create a GIF from the saved images
    with imageio.get_writer(gif_filename, mode='I', duration=1) as writer:
        for filename in images:
            image = imageio.imread(filename)
            writer.append_data(image)