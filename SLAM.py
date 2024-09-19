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

# Define desired figure size
FIG_WIDTH = 1200  # in pixels
FIG_HEIGHT = 800  # in pixels

# Run Monte Carlo simulations
for mcc in range(1, num_MC + 1):

    ###############################################################
    # Initialization of Simulation Parameters and Variables
    ###############################################################

    # Initialize Base Station (BS) and Virtual Anchors (VA)
    BS = np.array([[0], [0], [40]])  # Base Station coordinates (x, y, z)
    VA = np.array([
        [200, 10, 20],
        [-200, -10, 20],
        [10, 200, 10],
        [-10, -200, 10]
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
    measurement1, v1real_state, index, pos_s_all, association_record, clutter_positions_all = create_measurement(
        s1_in, K, s1_v, s1_ang_v, T, R, BS, VA, SP
    )

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
            marker=dict(size=5, color='red'),  # Increased marker size for better visibility
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
            marker=dict(size=3, color='blue'),
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
            marker=dict(size=3, color='green'),
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
            line=dict(color='black', width=3),
            marker=dict(size=4, color='black'),
            name='User Trajectory'
        ))

        # Plot user's current position
        user_position = np.array([v1real_state[0, step], v1real_state[1, step], 0])
        fig.add_trace(go.Scatter3d(
            x=[user_position[0]],
            y=[user_position[1]],
            z=[user_position[2]],
            mode='markers+text',
            marker=dict(size=7, color='black'),
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
            sizeref=15,  # Adjust sizeref to change the arrow size
            anchor="tail",
            name='User Heading'
        ))

        # Plot measurements at current step
        z_k = measurement1[step]               # Measurements at current step (list of arrays)
        index_s = index[step]                  # Indices of measurement sources (list)
        pos_s_list = pos_s_all[step]           # pos_s for measurements at this time step
        assoc_step = association_record[step]   # Associations for this step
        clutter_positions = clutter_positions_all[step]  # Clutter positions for this step

        # Debugging: Print lengths to ensure synchronization
        print(f"Step {step+1}: len(z_k) = {len(z_k)}, len(pos_s_list) = {len(pos_s_list)}, len(clutter_positions) = {len(clutter_positions)}")

        # Initialize legend flags for the current step
        reflection_path_shown = False
        reflection_point_shown = False
        AoA_shown = False
        AoD_shown = False

        # For each measurement, plot the AoA and AoD lines based on association
        for j in range(len(z_k)):
            # Ensure pos_s_list has enough entries
            if j >= len(pos_s_list):
                print(f"Error: pos_s_list has only {len(pos_s_list)} entries, but trying to access index {j}")
                continue  # Skip this measurement

            # Extract measurement data
            z = z_k[j]
            range_ = z[0]          # Range (TOA)
            azimuth_AoA = z[3] + user_heading   # Adjust AoA azimuth with user's heading
            elevation_AoA = z[4]   # Elevation AoA
            idx = index_s[j]       # Measurement source index
            pos_s = pos_s_list[j]  # pos_s for this measurement (None if not VA)
            assoc = assoc_step[j]   # Association tuple ('BS', j), ('VA', j), etc.

            source_type, source_idx = assoc  # Unpack association

            if source_type == 'clutter':
                # Plot only a black dot for clutter using stored clutter positions
                if j < len(clutter_positions):
                    clutter_pos = clutter_positions[j]
                    clutter_x, clutter_y, clutter_z = clutter_pos
                else:
                    print(f"Warning: No clutter position available for measurement {j} at step {step+1}")
                    clutter_x = clutter_y = clutter_z = 0  # Default position

                fig.add_trace(go.Scatter3d(
                    x=[clutter_x],
                    y=[clutter_y],
                    z=[clutter_z],
                    mode='markers',
                    marker=dict(size=4, color='black'),
                    name='Clutter' if j == 0 else None,
                    showlegend=False if j > 0 else True  # Show legend only once
                ))
                continue  # Skip plotting lines for clutter

            # Determine source position based on source type
            if source_type == 'BS':
                source_pos = BS[:, source_idx].flatten()
            elif source_type == 'VA':
                source_pos = VA[source_idx, :].flatten()
            elif source_type == 'SP':
                source_pos = SP[:, source_idx].flatten()
            else:
                source_pos = None  # Undefined source

            if source_pos is not None:
                # Plot AoD line from source to user
                fig.add_trace(go.Scatter3d(
                    x=[source_pos[0], user_position[0]],
                    y=[source_pos[1], user_position[1]],
                    z=[source_pos[2], user_position[2]],
                    mode='lines',
                    line=dict(color='orange', dash='dot', width=3),
                    showlegend=False if AoD_shown else True,
                    name='AoD Line' if not AoD_shown else None
                ))
                if not AoD_shown:
                    AoD_shown = True  # Set flag after first AoD line

            # Maintain reflection and AoA plotting
            if source_type == 'VA' and pos_s is not None:
                AoA_end = pos_s  # The AoA line ends at pos_s
                # Plot reflection path from BS to pos_s
                fig.add_trace(go.Scatter3d(
                    x=[BS[0, 0], pos_s[0]],
                    y=[BS[1, 0], pos_s[1]],
                    z=[BS[2, 0], pos_s[2]],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=3),
                    showlegend=False if reflection_path_shown else True,
                    name='Reflection Path' if not reflection_path_shown else None
                ))
                if not reflection_path_shown:
                    reflection_path_shown = True  # Set flag after first reflection path

                # Plot reflection path from pos_s to user
                fig.add_trace(go.Scatter3d(
                    x=[pos_s[0], user_position[0]],
                    y=[pos_s[1], user_position[1]],
                    z=[pos_s[2], user_position[2]],
                    mode='lines',
                    line=dict(color='red', dash='dash', width=3),
                    showlegend=False
                ))

                # Mark pos_s
                fig.add_trace(go.Scatter3d(
                    x=[pos_s[0]],
                    y=[pos_s[1]],
                    z=[pos_s[2]],
                    mode='markers+text',
                    marker=dict(size=4, color='magenta', symbol='x'),
                    text=['Reflection Point'],
                    textposition='top center',
                    name='Reflection Point' if not reflection_point_shown else None,
                    showlegend=False if reflection_point_shown else True
                ))
                if not reflection_point_shown:
                    reflection_point_shown = True  # Set flag after first reflection point

                # Plot AoA line based on azimuth and elevation
                u_AoA = np.array([
                    np.cos(elevation_AoA) * np.cos(azimuth_AoA),
                    np.cos(elevation_AoA) * np.sin(azimuth_AoA),
                    np.sin(elevation_AoA)
                ])
                AoA_end_pos = user_position + u_AoA * range_

                # Plot AoA line from user
                fig.add_trace(go.Scatter3d(
                    x=[user_position[0], AoA_end_pos[0]],
                    y=[user_position[1], AoA_end_pos[1]],
                    z=[user_position[2], AoA_end_pos[2]],
                    mode='lines',
                    line=dict(color='black', dash='dash', width=4),
                    showlegend=False if AoA_shown else True,
                    name='AoA Line' if not AoA_shown else None
                ))
                if not AoA_shown:
                    AoA_shown = True  # Set flag after first AoA line

                # Mark AoA end point
                fig.add_trace(go.Scatter3d(
                    x=[AoA_end_pos[0]],
                    y=[AoA_end_pos[1]],
                    z=[AoA_end_pos[2]],
                    mode='markers',
                    marker=dict(size=4, color='black'),
                    name='AoA End' if not AoA_shown else None,
                    showlegend=False
                ))

        # Update layout with increased figure size
        fig.update_layout(
            width=FIG_WIDTH,    # Set the width of the figure
            height=FIG_HEIGHT,  # Set the height of the figure
            scene=dict(
                xaxis=dict(range=[-250, 250], title='X'),
                yaxis=dict(range=[-250, 250], title='Y'),
                zaxis=dict(range=[0, 50], title='Z'),
                aspectratio=dict(x=1, y=1, z=0.5),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.0)  # Adjust camera position for better view
                )
            ),
            title=f'Step {step + 1}',
            showlegend=True
        )

        # Save the figure as an HTML file
        html_filename = os.path.join(image_dir, f'frame_{step:03d}.html')
        pio.write_html(fig, file=html_filename, auto_open=False)

        # Save the figure as a PNG image with specified dimensions
        png_filename = os.path.join(image_dir, f'frame_{step:03d}.png')
        fig.write_image(png_filename, width=FIG_WIDTH, height=FIG_HEIGHT, scale=2)  # scale=2 for higher resolution
        images.append(png_filename)

    # Create a GIF from the saved images
    with imageio.get_writer(gif_filename, mode='I', duration=1) as writer:
        for filename in images:
            image = imageio.imread(filename)
            writer.append_data(image)

    print(f"Monte Carlo simulation completed. GIF saved as {gif_filename}.")
