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

    # Initialize legend flags outside the step loop to add legend entries only once
    clutter_legend_added = False
    sp_meas_legend_added = False

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

        # Plot user's trajectory up to current step in shiny green
        fig.add_trace(go.Scatter3d(
            x=v1real_state[0, :step+1],
            y=v1real_state[1, :step+1],
            z=np.zeros(step+1),
            mode='lines+markers',
            line=dict(color='limegreen', width=3),
            marker=dict(size=4, color='limegreen'),
            name='User Trajectory'
        ))

        # Plot user's current position in shiny green
        user_position = np.array([v1real_state[0, step], v1real_state[1, step], 0])
        fig.add_trace(go.Scatter3d(
            x=[user_position[0]],
            y=[user_position[1]],
            z=[user_position[2]],
            mode='markers+text',
            marker=dict(size=7, color='limegreen'),
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
            colorscale=[[0, 'limegreen'], [1, 'limegreen']],
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

        # Initialize lists to collect SP measurements and clutter points
        sp_meas_x = []
        sp_meas_y = []
        sp_meas_z = []
        clutter_meas_x = []
        clutter_meas_y = []
        clutter_meas_z = []

        # For each measurement, categorize and collect positions
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
                # Collect clutter positions
                clutter_pos = clutter_positions[source_idx]
                clutter_x, clutter_y, clutter_z = clutter_pos

                clutter_meas_x.append(clutter_x)
                clutter_meas_y.append(clutter_y)
                clutter_meas_z.append(clutter_z)
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

            if source_pos is not None and source_type != 'SP':
                # Plot AoD line from source to user (excluding SP to avoid cluttering)
                fig.add_trace(go.Scatter3d(
                    x=[source_pos[0], user_position[0]],
                    y=[source_pos[1], user_position[1]],
                    z=[source_pos[2], user_position[2]],
                    mode='lines',
                    line=dict(color='orange', dash='dot', width=3),
                    name='AoD Line'
                ))

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
                    name='Reflection Path'
                ))

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
                    marker=dict(size=2, color='magenta', symbol='x'),
                    text=['Reflection Point'],
                    textposition='top center',
                    name='Reflection Point'
                ))

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
                    name='AoA Line'
                ))

                # Mark AoA end point
                fig.add_trace(go.Scatter3d(
                    x=[AoA_end_pos[0]],
                    y=[AoA_end_pos[1]],
                    z=[AoA_end_pos[2]],
                    mode='markers',
                    marker=dict(size=4, color='black'),
                    name='AoA End',
                    showlegend=False
                ))

            # Collect SP measurements for plotting
            if source_type == 'SP' and source_type is not None:
                # Calculate measurement position based on range and angles
                # Assuming elevation_AoA is in radians
                x_meas = user_position[0] + range_ * np.cos(elevation_AoA) * np.cos(azimuth_AoA)
                y_meas = user_position[1] + range_ * np.cos(elevation_AoA) * np.sin(azimuth_AoA)
                z_meas = user_position[2] + range_ * np.sin(elevation_AoA)
                sp_meas_x.append(x_meas)
                sp_meas_y.append(y_meas)
                sp_meas_z.append(z_meas)

        # After processing all measurements, plot SP measurements and clutter

        # Plot SP Measurements as black circles
        if sp_meas_x and sp_meas_y and sp_meas_z:
            fig.add_trace(go.Scatter3d(
                x=sp_meas_x,
                y=sp_meas_y,
                z=sp_meas_z,
                mode='markers',
                marker=dict(size=4, color='black', symbol='circle'),
                name='SP Measurements'
            ))

        # Plot Clutter Measurements as black crosses
        if clutter_meas_x and clutter_meas_y and clutter_meas_z:
            fig.add_trace(go.Scatter3d(
                x=clutter_meas_x,
                y=clutter_meas_y,
                z=clutter_meas_z,
                mode='markers',
                showlegend=True,
                marker=dict(size=2, color='black', symbol='x'),
                name='Clutter'
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

    # Optional: Clean up temporary PNG files to save disk space
    # for filename in images:
    #     os.remove(filename)
    # print("Temporary image files deleted.")
