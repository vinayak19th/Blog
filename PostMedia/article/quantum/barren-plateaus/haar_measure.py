import pennylane as qp
import numpy as np
import plotly.graph_objects as go

# set the random seed
np.random.seed(42)

# Use the mixed state simulator to save some steps in plotting later
dev = qp.device('default.mixed', wires=1)

@qp.qnode(dev)
def not_a_haar_random_unitary():
    # Sample all parameters from their flat uniform distribution
    phi, theta, omega = 2 * np.pi * np.random.uniform(size=3)
    qp.Rot(phi, theta, omega, wires=0)
    return qp.state()

def plot_bloch_sphere(bloch_vectors):
    """ Helper function to plot vectors on a sphere."""
    colors = {
        'points': '#e29d9e',
        'text': '#000000',
        'axes': '#000000'
    }

    fig = go.Figure()

    # Draw the axes
    # x axis
    fig.add_trace(go.Scatter3d(x=[-1.5, 1.5], y=[0, 0], z=[0, 0], mode='lines', line=dict(color=colors['axes'], width=2), showlegend=False))
    # y axis
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1.5, 1.5], z=[0, 0], mode='lines', line=dict(color=colors['axes'], width=2), showlegend=False))
    # z axis
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1.5, 1.5], mode='lines', line=dict(color=colors['axes'], width=2), showlegend=False))

    # Add text labels
    fig.add_trace(go.Scatter3d(
        x=[0, 0, 1.25, -1.25, 0, 0],
        y=[0, 0, 0, 0, 1.25, -1.25],
        z=[1.25, -1.25, 0, 0, 0, 0],
        mode='text',
        text=[r"|0⟩", r"|1⟩", r"|+⟩", r"|–⟩", r"|i+⟩", r"|i–⟩"],
        textfont=dict(color=colors['text'], size=16),
        showlegend=False
    ))

    # Add scatter points
    fig.add_trace(go.Scatter3d(
        x=bloch_vectors[:,0], 
        y=bloch_vectors[:,1], 
        z=bloch_vectors[:,2], 
        mode='markers', 
        marker=dict(size=3, color=colors['points'], opacity=0.3),
        showlegend=False
    ))

    # Configure layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False, zeroline=False),
            yaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False, zeroline=False),
            zaxis=dict(showbackground=False, showticklabels=False, title='', showgrid=False, zeroline=False),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.1, y=1.1, z=1.1)  # Closer default zoom
            )
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=0),
        width=600,
        height=600
    )

    fig.show(config={'scrollZoom': False, 'displayModeBar': False})

num_samples = 2021

not_haar_samples = [not_a_haar_random_unitary() for _ in range(num_samples)]

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

# Used the mixed state simulator so we could have the density matrix for this part!
def convert_to_bloch_vector(rho):
    """Convert a density matrix to a Bloch vector."""
    ax = np.trace(np.dot(rho, X)).real
    ay = np.trace(np.dot(rho, Y)).real
    az = np.trace(np.dot(rho, Z)).real
    return [ax, ay, az]

not_haar_bloch_vectors = np.array([convert_to_bloch_vector(s) for s in not_haar_samples])

plot_bloch_sphere(not_haar_bloch_vectors)