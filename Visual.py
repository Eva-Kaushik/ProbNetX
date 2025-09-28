import networkx as nx
import plotly.graph_objects as go
import plotly.io as pio

edges = [
    ("3d", "spam"), ("meeting", "spam"), 
    ("spam", "business"), ("spam", "george"),
    ("spam", "lab"), ("george", "lab"), ("meeting", "lab"),
    ("spam", "hp"), ("lab", "hp"), ("business", "hp"),
    ("spam", "internet"), ("3d", "internet"), ("business", "internet"), ("hp", "internet"),
    ("spam", "dollarsign"), ("business", "dollarsign"), ("internet", "dollarsign"),
    ("spam", "000"), ("dollarsign", "000"),
    ("hp", "1999"), ("3d", "1999"), ("000", "1999"),
    ("spam", "receive"), ("000", "receive"), ("internet", "receive"), ("1999", "receive"),
    ("spam", "remove"), ("receive", "remove"),
    ("spam", "free"), ("receive", "free"), ("1999", "free"),
    ("spam", "edu"), ("receive", "edu"), ("george", "edu"), ("hp", "edu"),
    ("george", "squarebracket"), ("1999", "squarebracket"), ("edu", "squarebracket")
]

G = nx.DiGraph()
G.add_edges_from(edges)

spam_node = ["spam"]
node_colors = ['#FF6F61' if n in spam_node else '#6FA8DC' for n in G.nodes()]

pos = nx.spring_layout(G, seed=42, k=1.5)
edge_x = []
edge_y = []
for e in G.edges():
    x0, y0 = pos[e[0]]
    x1, y1 = pos[e[1]]
    edge_x += [x0, x1, None]
    edge_y += [y0, y1, None]

edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=1.5, color='gray'),
    hoverinfo='none',
    mode='lines'
)

node_x = []
node_y = []
hover_text = []
for n in G.nodes():
    x, y = pos[n]
    node_x.append(x)
    node_y.append(y)
    hover_text.append(f"{n}<br>X: {x:.2f}<br>Y: {y:.2f}")

node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=[n for n in G.nodes()],
    textposition='bottom center',
    hovertext=hover_text,
    hoverinfo='text',
    marker=dict(size=25, color=node_colors)
)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title='ProbNetX Bayesian Network - Spam Detection',
                    title_x=0.5,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                ))

fig.write_html("ProbNetX_interactive.html")
print("Interactive HTML saved as ProbNetX_interactive.html")

pio.write_image(fig, "ProbNetX_graph.png", scale=3)
print("Static PNG saved as ProbNetX_graph.png")

fig.show()

