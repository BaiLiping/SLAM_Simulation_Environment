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
FIG_WIDTH = 2000  # in pixels
FIG_HEIGHT = 1500  # in pixels

# Define legend entries with their corresponding styles
legend_entries = [
    {
        'name': 'BS',
        'type': 'Scatter3d',
        'mode': 'markers+text',
        'marker': {'size': 7, 'color': 'red'},
        'text': ['BS'],
        'textposition': 'top center'
    },
    {
        'name': 'VA',
        'type': 'Scatter3d',
        'mode': 'markers+text',
        'marker': {'size': 7, 'color': 'purple'},
        'text': [f'VA{va_idx+1}' for va_idx in range(4)],
        'textposition': 'top center'
    },
    {
        'name': 'SP',
        'type': 'Scatter3d',
        'mode': 'markers+text',
        'marker': {'size': 7, 'color': 'green'},
        'text': [f'SP{sp_idx+1}' for sp_idx in range(4)],
        'textposition': 'top center'
    },
    {
        'name': 'User Trajectory',
        'type': 'Scatter3d',
        'mode': 'lines+markers',
        'line': {'color': 'limegreen', 'width': 3},
        'marker': {'size': 4, 'color': 'limegreen'}
    },
    {
        'name': 'User Position',
        'type': 'Scatter3d',
        'mode': 'markers+text',
        'marker': {'size': 7, 'color': 'limegreen'},
        'text': ['User'],
        'textposition': 'top center'
    },
    {
        'name': 'User Heading',
        'type': 'Cone',
        'marker': {'size': 7, 'color': 'limegreen'}
    },
    {
        'name': 'AoD Line',
        'type': 'Scatter3d',
        'mode': 'lines',
        'line': {'color': 'orange', 'dash': 'dot', 'width': 3}
    },
    {
        'name': 'AoD End',
        'type': 'Scatter3d',
        'mode': 'markers',
        'marker': {'size': 7, 'color': 'orange'},
        'text': ['AoD End'],
        'textposition': 'top center'
    },
    {
        'name': 'AoA Line',
        'type': 'Scatter3d',
        'mode': 'lines',
        'line': {'color': 'blue', 'dash': 'dash', 'width': 4}
    },
    {
        'name': 'AoA End',
        'type': 'Scatter3d',
        'mode': 'markers',
        'marker': {'size': 7, 'color': 'blue'},
        'text': ['AoA End'],
        'textposition': 'top center'
    },
    {
        'name': 'Reflection Path',
        'type': 'Scatter3d',
        'mode': 'lines',
        'line': {'color': 'red', 'dash': 'dash', 'width': 3}
    },
    {
        'name': 'Reflection Point',
        'type': 'Scatter3d',
        'mode': 'markers+text',
        'marker': {'size': 7, 'color': 'magenta', 'symbol': 'x'},
        'text': ['RP'],
        'textposition': 'top center'
    },
    {
        'name': 'Clutter',
        'type': 'Scatter3d',
        'mode': 'markers',
        'marker': {'size': 7, 'color': 'black', 'symbol': 'x'}
    }
]

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

    # Initial state of the user [x, y, z, heading, bias]
    s1_in = np.array([70.7285, 0, 0, np.pi / 2, 300])  # Adjusted to include z-coordinate
    s1_v = 22.22       # User speed (m/s)
    s1_ang_v = np.pi / 10  # User angular velocity (rad/s)

    ###############################################################
    # Measurement Generation
    ###############################################################

    # Updated to unpack pos_vb_all
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
            marker=dict(size=7, color='red'),
            text=['BS'],
            textposition='top center',
            name='BS',
            showlegend=False
        ))

        # Plot Virtual Anchors (VA)
        fig.add_trace(go.Scatter3d(
            x=VA[:, 0],
            y=VA[:, 1],
            z=VA[:, 2],
            mode='markers+text',
            marker=dict(size=7, color='purple'),
            text=[f'VA{va_idx+1}' for va_idx in range(VA.shape[0])],
            textposition='top center',
            name='VA',
            showlegend=False
        ))

        # Plot Scatter Points (SP)
        fig.add_trace(go.Scatter3d(
            x=SP[:, 0],
            y=SP[:, 1],
            z=SP[:, 2],
            mode='markers+text',
            marker=dict(size=7, color='green'),
            text=[f'SP{sp_idx+1}' for sp_idx in range(SP.shape[0])],
            textposition='top center',
            name='SP',
            showlegend=False
        ))

        # Plot user's trajectory up to current step in shiny green
        fig.add_trace(go.Scatter3d(
            x=v1real_state[0, :step+1],
            y=v1real_state[1, :step+1],
            z=v1real_state[2, :step+1],  # Use actual z-coordinates
            mode='lines+markers',
            line=dict(color='limegreen', width=3),
            marker=dict(size=4, color='limegreen'),
            name='User Trajectory',
            showlegend=False
        ))

        # Plot user's current position in shiny green
        user_position = np.array([v1real_state[0, step], v1real_state[1, step], v1real_state[2, step]])
        fig.add_trace(go.Scatter3d(
            x=[user_position[0]],
            y=[user_position[1]],
            z=[user_position[2]],
            mode='markers+text',
            marker=dict(size=7, color='limegreen'),
            text=['User'],
            textposition='top center',
            name='User Position',
            showlegend=False
        ))

        # Plot an arrow indicating the user's heading
        user_heading = v1real_state[3, step]  # User's heading at current step
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
            name='User Heading',
            showlegend=False
        ))

        # Plot measurements at current step
        z_k = measurement1[step]               # Measurements at current step (list of arrays)
        index_s = index[step]                  # Indices of measurement sources (list)
        pos_s_list = pos_s_all[step]           # pos_s for measurements at this time step
        assoc_step = association_record[step]   # Associations for this step
        clutter_positions = clutter_positions_all[step]  # Clutter positions for this step

        # Debugging: Print lengths to ensure synchronization
        print(f"Step {step+1}: len(z_k) = {len(z_k)}, len(pos_s_list) = {len(pos_s_list)}, len(clutter_positions) = {len(clutter_positions)}")

        # Initialize a counter for SP measurements to map to pos_vb_step
        sp_counter = 0

        # Loop through each measurement
        for j in range(len(z_k)):
            # Ensure pos_s_list has enough entries
            if j >= len(pos_s_list):
                print(f"Error: pos_s_list has only {len(pos_s_list)} entries, but trying to access index {j}")
                continue  # Skip this measurement

            # Extract measurement data
            z = z_k[j]
            range_ = z[0]          # Range (TOA)
            azimuth_AoD = z[1]   
            elevation_AoD = z[2]   # Elevation AoD
            azimuth_AoA = z[3] + user_heading   # Adjust AoA azimuth with user's heading
            elevation_AoA = z[4]   # Elevation AoA
            idx = index_s[j]       # Measurement source index
            pos_s = pos_s_list[j]  # pos_s for this measurement (None if not VA)
            assoc = assoc_step[j]   # Association tuple ('BS', j), ('VA', j), etc.

            source_type, source_idx = assoc  # Unpack association

            # Determine if the measurement is from BS, VA, SP, or Clutter
            if source_type in ['BS', 'VA', 'SP']:
                # Retrieve source position
                if source_type == 'BS':
                    source_pos = BS[:, source_idx].flatten()
                elif source_type == 'VA':
                    source_pos = VA[source_idx, :].flatten()
                elif source_type == 'SP':
                    source_pos = SP[source_idx, :].flatten()

                # Compute AoD direction vector
                u_AoD = np.array([
                    np.cos(elevation_AoD) * np.cos(azimuth_AoD),
                    np.cos(elevation_AoD) * np.sin(azimuth_AoD),
                    np.sin(elevation_AoD)
                ])
                AoD_end_pos = BS.flatten() + u_AoD * range_  # Always start AoD from BS

                # Plot AoD line from BS to AoD end position
                fig.add_trace(go.Scatter3d(
                    x=[BS[0, 0], AoD_end_pos[0]],
                    y=[BS[1, 0], AoD_end_pos[1]],
                    z=[BS[2, 0], AoD_end_pos[2]],
                    mode='lines',
                    line=dict(color='orange', dash='dot', width=3),
                    name='AoD Line',
                    showlegend=False
                ))

                # Optionally, mark AoD end point
                fig.add_trace(go.Scatter3d(
                    x=[AoD_end_pos[0]],
                    y=[AoD_end_pos[1]],
                    z=[AoD_end_pos[2]],
                    mode='markers',
                    marker=dict(size=4, color='orange'),
                    name='AoD End',
                    showlegend=False
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
                    line=dict(color='blue', dash='dash', width=4),
                    name='AoA Line',
                    showlegend=False
                ))

                # Mark AoA end point
                fig.add_trace(go.Scatter3d(
                    x=[AoA_end_pos[0]],
                    y=[AoA_end_pos[1]],
                    z=[AoA_end_pos[2]],
                    mode='markers',
                    marker=dict(size=4, color='blue'),
                    name='AoA End',
                    showlegend=False
                ))

                # Handle Reflection Path for 'VA' and 'SP' sources
                if source_type in ['VA']:
                    # Plot reflection path from BS to pos_s
                    fig.add_trace(go.Scatter3d(
                        x=[BS[0, 0], pos_s[0]],
                        y=[BS[1, 0], pos_s[1]],
                        z=[BS[2, 0], pos_s[2]],
                        mode='lines',
                        line=dict(color='red', dash='dash', width=3),
                        name='Reflection Path',
                        showlegend=False
                    ))

                    # Plot reflection path from pos_s to user
                    fig.add_trace(go.Scatter3d(
                        x=[pos_s[0], user_position[0]],
                        y=[pos_s[1], user_position[1]],
                        z=[pos_s[2], user_position[2]],
                        mode='lines',
                        line=dict(color='red', dash='dash', width=3),
                        name='Reflection Path',
                        showlegend=False
                    ))

                    # Mark pos_s
                    fig.add_trace(go.Scatter3d(
                        x=[pos_s[0]],
                        y=[pos_s[1]],
                        z=[pos_s[2]],
                        mode='markers+text',
                        marker=dict(size=2, color='magenta', symbol='x'),
                        text=['RP'],
                        textposition='top center',
                        name='Reflection Point',
                        showlegend=False
                    ))

        # Plot clutter measurements (without AoA and AoD lines)
        if len(clutter_positions) > 0:
            clutter_meas_x = [clutter[0] for clutter in clutter_positions]
            clutter_meas_y = [clutter[1] for clutter in clutter_positions]
            clutter_meas_z = [clutter[2] for clutter in clutter_positions]
            fig.add_trace(go.Scatter3d(
                x=clutter_meas_x,
                y=clutter_meas_y,
                z=clutter_meas_z,
                mode='markers',
                marker=dict(size=2, color='black', symbol='x'),
                name='Clutter',
                showlegend=False
            ))

        # Add dummy traces for legend entries
        for entry in legend_entries:
            if entry['type'] == 'Cone':
                # For cones, add a representative cone for the legend
                # Position it outside the plot range to avoid visibility
                fig.add_trace(go.Cone(
                    x=[1000],  # Outside the plot range
                    y=[1000],
                    z=[1000],
                    u=[1],      # Arbitrary direction
                    v=[1],
                    w=[1],
                    colorscale=[[0, entry['marker']['color']], [1, entry['marker']['color']]],
                    showscale=False,
                    sizemode='absolute',
                    sizeref=15,  # Adjust sizeref to change the arrow size
                    anchor="tail",
                    name=entry['name'],
                    showlegend=True
                ))
            elif entry['type'] == 'Scatter3d':
                # For Scatter3d, add a dummy point outside the plot range
                dummy_x = [1000]  # Outside the plot range
                dummy_y = [1000]
                dummy_z = [1000]
                scatter_args = {
                    'x': dummy_x,
                    'y': dummy_y,
                    'z': dummy_z,
                    'mode': entry['mode'],
                    'marker': entry.get('marker', {}),
                    'line': entry.get('line', {}),
                    'text': entry.get('text', []),
                    'name': entry['name'],
                    'showlegend': True
                }
                # Only set 'textposition' if it's defined
                if 'textposition' in entry:
                    scatter_args['textposition'] = entry['textposition']
                fig.add_trace(go.Scatter3d(**scatter_args))

        # Update layout with increased figure size
        fig.update_layout(
            width=FIG_WIDTH,    # Set the width of the figure
            height=FIG_HEIGHT,  # Set the height of the figure
            scene=dict(
                xaxis=dict(range=[-350, 350], title='X'),
                yaxis=dict(range=[-350, 350], title='Y'),
                zaxis=dict(range=[-50, 50], title='Z'),
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
