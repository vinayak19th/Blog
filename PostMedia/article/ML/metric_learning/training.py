"""
Metric Learning Traing Animation using Plotly.
Visualizes the forward pass of a Triplet (Anchor, Positive, Negative) through an encoder
and their corresponding distance updates in the latent space over training epochs.
"""

import plotly.graph_objects as go
import base64
import os

def encode_image(img_name: str) -> str:
    """Encodes an image to a base64 string for Plotly layout.images."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(script_dir, img_name)
    if not os.path.exists(img_path):
        return ""
    with open(img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/png;base64,{encoded_string}"

def create_animation():
    """Generates the Metric Learning Plotly animation."""
    # Base64 encode images
    anchor_base64 = encode_image("anchor.png")
    pos_base64 = encode_image("positive.png")
    neg_base64 = encode_image("Negative.png")

    # Animation keyframes
    epochs = [0, 5, 25, 50]

    # Latent space coordinates over epochs
    # Make epoch 0 starting state much more scattered, but keep x > 4.5 and y < 9.0
    latent_anchor = (7.0, 5.0)
    latent_points = {
        0:  {'pos': (5.0, 8.5), 'neg': (7.0, 1.5)},
        5:  {'pos': (6.0, 7.0), 'neg': (7.5, 3.5)},
        25: {'pos': (6.6, 5.6), 'neg': (8.5, 5.0)},
        50: {'pos': (6.8, 5.2), 'neg': (9, 6.5)}
    }

    # Static elements on the left side (Model Architecture)
    shapes = [
        # Encoder Trapezium
        dict(
            type="path",
            # A trapezium rotated 90 degrees
            path="M 2.5,2.0 L 2.5,8.0 L 3.5,6.5 L 3.5,3.5 Z",
            fillcolor="rgba(100, 150, 250, 0.5)",
            line=dict(color="#1f77b4", width=2),
        ),
        # Connecting arrows (Images to Encoder)
        dict(type="line", x0=1.5, y0=8.0, x1=2.5, y1=6.5, line=dict(color="#1f77b4", width=2, dash="dash")), # Anchor
        dict(type="line", x0=1.5, y0=5.0, x1=2.5, y1=5.0, line=dict(color="#2ca02c", width=2, dash="dash")), # Positive
        dict(type="line", x0=1.5, y0=2.0, x1=2.5, y1=3.5, line=dict(color="#d62728", width=2, dash="dash")), # Negative
        # Divider Line
        dict(type="line", x0=4.5, y0=0.0, x1=4.5, y1=10.0, line=dict(color="black", width=2))
    ]

    images = [
        dict(source=anchor_base64, x=0.5, y=8.75, sizex=1.5, sizey=1.5, xref="x", yref="y", xanchor="left", yanchor="top", layer="above"),
        dict(source=pos_base64, x=0.5, y=5.75, sizex=1.5, sizey=1.5, xref="x", yref="y", xanchor="left", yanchor="top", layer="above"),
        dict(source=neg_base64, x=0.5, y=2.75, sizex=1.5, sizey=1.5, xref="x", yref="y", xanchor="left", yanchor="top", layer="above")
    ]

    # Base Static Annotations
    base_annotations = [
        dict(x=2.25, y=9.5, text="<b>Model Architecture</b>", showarrow=False, font=dict(size=18), xanchor="center"),
        dict(x=7.25, y=9.5, text="<b>Latent Space</b>", showarrow=False, font=dict(size=18), xanchor="center"),
        dict(x=0.5, y=7.0, text="Anchor<br>Input", showarrow=False, font=dict(size=12), xanchor="left", yanchor="top"),
        dict(x=0.5, y=4.0, text="Positive<br>Input", showarrow=False, font=dict(size=12), xanchor="left", yanchor="top"),
        dict(x=0.5, y=1.0, text="Negative<br>Input", showarrow=False, font=dict(size=12), xanchor="left", yanchor="top"),
        dict(x=3.0, y=5.0, text="<b>Encoder</b><br><i>(CNN/ViT)</i>", showarrow=False, font=dict(size=12, color="black"), xanchor="center", yanchor="middle")
    ]

    # Generate Frames
    frames = []
    # Need to keep track of the names for the slider steps
    frame_names = []
    
    import math
    def dist(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)*5

    for ep in epochs:
        pos = latent_points[ep]['pos']
        neg = latent_points[ep]['neg']
        
        # Calculate distances for the loss frame
        d_ap = dist(latent_anchor, pos)
        d_an = dist(latent_anchor, neg)
        loss_val = max(d_ap - d_an + 5, (d_an-d_ap)*0.01) # margin = 1.0
        
        # Midpoints for labels
        mid_ap_x, mid_ap_y = (latent_anchor[0] + pos[0]) / 2, (latent_anchor[1] + pos[1]) / 2
        mid_an_x, mid_an_y = (latent_anchor[0] + neg[0]) / 2, (latent_anchor[1] + neg[1]) / 2
        
        # Projection lines from encoder (3.5, 5.0) to latent points (Color-matched)
        proj_lines = [
            go.Scatter(x=[3.5, latent_anchor[0]], y=[5.0, latent_anchor[1]], mode='lines', line=dict(color='#1f77b4', width=2, dash='dash'), showlegend=False),
            go.Scatter(x=[3.5, pos[0]], y=[5.0, pos[1]], mode='lines', line=dict(color='#2ca02c', width=2, dash='dash'), showlegend=False),
            go.Scatter(x=[3.5, neg[0]], y=[5.0, neg[1]], mode='lines', line=dict(color='#d62728', width=2, dash='dash'), showlegend=False),
        ]
        
        # Base markers
        markers = [
            go.Scatter(
                x=[latent_anchor[0]], y=[latent_anchor[1]], mode='markers+text', 
                marker=dict(size=18, color='#1f77b4', line=dict(width=2, color='white')),
                text=["Anchor"], textposition="bottom center", name='Anchor', showlegend=True
            ),
            go.Scatter(
                x=[pos[0]], y=[pos[1]], mode='markers+text', 
                marker=dict(size=18, color='#2ca02c', line=dict(width=2, color='white')),
                text=["Positive"], textposition="bottom center", name='Positive', showlegend=True
            ),
            go.Scatter(
                x=[neg[0]], y=[neg[1]], mode='markers+text', 
                marker=dict(size=18, color='#d62728', line=dict(width=2, color='white')),
                text=["Negative"], textposition="bottom center", name='Negative', showlegend=True
            )
        ]

        # Lines representing distance from Anchor to Pos/Neg
        scatter_x = [latent_anchor[0], pos[0], None, latent_anchor[0], neg[0]]
        scatter_y = [latent_anchor[1], pos[1], None, latent_anchor[1], neg[1]]
        
        # --- Standard Frame ---
        frame_data = proj_lines + [
            go.Scatter(
                x=scatter_x, y=scatter_y, mode='lines', 
                line=dict(color='black', width=1, dash='dot'), showlegend=False
            )
        ] + markers + [
            # Dummy trace for loss_annotations_trace (No d(A,P) text, but maintain trace structure for Plotly)
            go.Scatter(x=[mid_ap_x, mid_an_x], y=[mid_ap_y, mid_an_y], mode='text', text=["", ""], showlegend=False),
            # Keep the explicit loss bar visible
            go.Scatter(x=[9.0, 9.6, 9.6, 9.0, 9.0], y=[0, 0, loss_val, loss_val, 0], fill="toself", fillcolor="rgba(255, 0, 0, 0.5)", line=dict(color="red", width=2), mode='lines', showlegend=False),
            # Keep the explicit loss value text visible
            go.Scatter(x=[9.3], y=[loss_val + 0.4], mode='text', text=[f"<b>Loss: {loss_val:.2f}</b>"], textfont=dict(color="red", size=14), showlegend=False)
        ]
        name_std = f"Epoch {ep}"
        
        # Plotly explicitly needs us to overwrite the extra annotations from the Loss frame
        # by providing invisible annotations at those indices (7th and 8th)
        clear_arrows = [
            dict(text="", showarrow=False, opacity=0),
            dict(text="", showarrow=False, opacity=0),
            dict(text="", showarrow=False, opacity=0),
            dict(text="", showarrow=False, opacity=0)
        ]
        frames.append(go.Frame(data=frame_data, name=name_std, layout=go.Layout(annotations=base_annotations + clear_arrows)))
        frame_names.append(name_std)
        
        # --- Loss Measurement Frame ---
        # Bolder, darker distance lines
        loss_lines = go.Scatter(
            x=scatter_x, y=scatter_y, mode='lines', 
            line=dict(color='black', width=3, dash='dot'), showlegend=False
        )
        
        # Loss Annotations (placed at midpoint of the lines)
        loss_annotations_trace = go.Scatter(
            x=[mid_ap_x, mid_an_x], 
            y=[mid_ap_y + 0.3, mid_an_y + 0.3], 
            mode='text',
            text=[f"d(A,P)={d_ap:.2f}", f"d(A,N)={d_an:.2f}"],
            textfont=dict(color=["#2ca02c", "#d62728"], size=[14, 14]),
            showlegend=False
        )
        
        # Loss bar visualization on the side (x=9.0 to 9.6)
        loss_bar = go.Scatter(
            x=[9.0, 9.6, 9.6, 9.0, 9.0],
            y=[0, 0, loss_val, loss_val, 0],
            fill="toself", fillcolor="rgba(255, 0, 0, 0.5)",
            line=dict(color="red", width=2),
            mode='lines', showlegend=False
        )
        loss_label = go.Scatter(
            x=[9.3], y=[loss_val + 0.4],
            mode='text', text=[f"<b>Loss: {loss_val:.2f}</b>"],
            textfont=dict(color="red", size=14), showlegend=False
        )
        
        def get_arrow_anno(p1, p2, color, text):
            # Calculate parallel offset pointing up-left or similar normal
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx**2 + dy**2)
            nx = -dy / length
            ny = dx / length
            
            gap = 0.4
            
            # Start and end points for the arrow line (parallel)
            x1 = p1[0] + gap * nx
            y1 = p1[1] + gap * ny
            x2 = p2[0] + gap * nx
            y2 = p2[1] + gap * ny
            
            arr = dict(
                x=x2, y=y2,
                ax=x1, ay=y1,
                xref='x', yref='y',
                axref='x', ayref='y',
                text='',
                showarrow=True,
                arrowhead=2,
                arrowside='start+end',
                arrowsize=1,
                arrowwidth=1.2,
                arrowcolor=color
            )
            
            # Text above the arrow
            gap_text = 0.7
            txt_x = (p1[0] + p2[0])/2 + gap_text * nx
            txt_y = (p1[1] + p2[1])/2 + gap_text * ny
            
            txt = dict(
                x=txt_x, y=txt_y,
                xref='x', yref='y',
                text=text,
                showarrow=False,
                font=dict(color=color, size=16)
            )
            
            return arr, txt
            
        ap_arr, ap_txt = get_arrow_anno(latent_anchor, pos, "black", "<b>AP</b>")
        an_arr, an_txt = get_arrow_anno(latent_anchor, neg, "black", "<b>AN</b>")
        arrow_annos = [ap_arr, ap_txt, an_arr, an_txt]
        
        frame_data_loss = proj_lines + [loss_lines] + markers + [loss_annotations_trace, loss_bar, loss_label]
        name_loss = f"Loss {ep}"
        frames.append(go.Frame(data=frame_data_loss, name=name_loss, layout=go.Layout(annotations=base_annotations + arrow_annos)))
        frame_names.append(name_loss)

    # Base Figure initialization (Using Epoch 0 data)
    fig = go.Figure(data=frames[0].data, frames=frames)

    # UI Controls: Play/Pause button and Animation Slider
    slider_steps = []
    for f_name in frame_names:
        step = dict(
            method="animate",
            args=[[f_name], dict(mode="immediate", transition=dict(duration=800), frame=dict(duration=800, redraw=False))],
            label=f_name
        )
        slider_steps.append(step)

    sliders = [dict(
        active=0,
        yanchor="top",
        xanchor="left",
        transition=dict(duration=800, easing="cubic-in-out"),
        pad=dict(b=10, t=50),
        len=0.6,
        x=0.2,
        y=0.15,
        steps=slider_steps
    )]

    fig.update_layout(
        xaxis=dict(range=[0, 10], showgrid=False, zeroline=False, visible=False),
        yaxis=dict(range=[0, 10], showgrid=False, zeroline=False, visible=False),
        images=images,
        shapes=shapes,
        annotations=base_annotations, # Explicitly only use base_annotations initially
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=1000, redraw=False), transition=dict(duration=1000), fromcurrent=True, mode="immediate")]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))])
            ],
            direction="left", pad=dict(r=10, t=25), showactive=False, x=0.18, xanchor="right", y=0.05, yanchor="top"
        )],
        sliders=sliders,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),
        width=1000,
        height=600, # Increased height to make room for legend at the top
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif")
    )

    # Save to file
    out_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metric_learning_animation.html")
    fig.write_html(out_file)
    
    # Inject CSS for flowing dashed lines
    with open(out_file, "a", encoding="utf-8") as f:
        f.write('''
<style>
/* Disable CSS animations so JS takes over completely with requestAnimationFrame for smooth constant speed */
path[stroke-dasharray], path[style*="stroke-dasharray"] {
    animation: none !important;
}
</style>
<script>
// Robust JS animation for Plotly shapes and scatter lines
let offset = 0;
let lastTime = performance.now();
const speed = 30; // pixels per second (constant speed)

function animateFlow(time) {
    let dt = (time - lastTime) / 1000;
    lastTime = time;
    
    // Decrease offset to move towards the encoder (depends on SVG path plot direction)
    offset -= speed * dt;
    // Wrap around to prevent precision issues over long times
    if (offset < -1000) offset += 1000;
    
    document.querySelectorAll('path').forEach(p => {
        let dash = p.style.strokeDasharray || p.getAttribute('stroke-dasharray');
        let stroke = p.style.stroke || p.getAttribute('stroke') || '';
        
        // Exclude dotted distance lines
        if (dash && dash !== 'none' && !dash.includes('2px') && !dash.includes('dot')) {
            // Target the blue, green, and red lines specifically
            if (stroke.includes('31, 119, 180') || stroke.includes('44, 160, 44') || stroke.includes('214, 39, 40') || 
                stroke.includes('#1f77b4') || stroke.includes('#2ca02c') || stroke.includes('#d62728')) {
                // Apply the negative offset as a string with 'px'
                p.style.strokeDashoffset = offset + 'px';
            }
        }
    });
    requestAnimationFrame(animateFlow);
}
requestAnimationFrame(animateFlow);
</script>
''')
    print(f"Animation saved to {out_file}")

if __name__ == "__main__":
    create_animation()
