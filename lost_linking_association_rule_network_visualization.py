import pandas as pd
import graphviz
from collections import defaultdict

df = pd.read_excel('core_rules_filtered_mode0.xlsx')

nodes = set()
edges = defaultdict(list)

for _, row in df.iterrows():
    antecedents = eval(row['antecedents'])
    consequents = eval(row['consequents'])
    confidence = row['confidence']

    for a in antecedents:
        for c in consequents:
            node_a = a.strip()
            node_c = c.strip()
            nodes.add(node_a)
            nodes.add(node_c)
            edges[(node_a, node_c)].append(confidence)

CONFIDENCE_BINS = {
    (0.5, 0.6): {'color': '#253494', 'label': 'Low (0.5–0.6)'},
    (0.6, 0.8): {'color': '#2c7fb8', 'label': 'Medium (0.6–0.8)'},
    (0.8, 1.0): {'color': '#7fcdbb', 'label': 'High (0.8–1.0)'}
}
NODE_STYLES = {
    'feature': {
        'shape': 'box',
        'style': 'filled,rounded',
        'fillcolor': '#E3F2FD',
        'color': '#1976D2'
    },
    'mode': {
        'shape': 'ellipse',
        'style': 'filled',
        'fillcolor': '#FFEBEE',
        'color': '#C62828'
    }
}

dot = graphviz.Digraph(
    name='Loan_LostLinking_Network',
    format='png',
    graph_attr={
        'rankdir': 'LR',
        'bgcolor': 'white',
        'fontname': 'Times New Roman',
        'fontsize': '14',
        'label': 'Association - rule network between customer features and lost - linking modes',
        'labelloc': 't',
        'labeljust': 'c',
        'labelfontsize': '12',
        'nodesep': '0.3',
        'ranksep': '0.6',
        'size': '16,9',
        'dpi': '300'
    },
    node_attr={
        'fontname': 'Times New Roman',
        'fontsize': '11',
        'height': '0.7',
        'width': '1.5'
    },
    edge_attr={
        'fontname': 'Times New Roman',
        'fontsize': '10',
        'arrowhead': 'normal'
    }
)

for node in nodes:
    if 'Lost - linking Mode' in node:
        dot.node(node, **NODE_STYLES['mode'])
    else:
        dot.node(node, **NODE_STYLES['feature'])

for (a, c), confidences in edges.items():
    avg_conf = sum(confidences) / len(confidences)
    for (low, high), style in CONFIDENCE_BINS.items():
        if low <= avg_conf < high:
            edge_color = style['color']
            break
    else:
        edge_color = '#9E9E9E'

    penwidth = str(1 + 2 * (avg_conf - 0.5))
    dot.edge(a, c, label=f"{avg_conf:.2f}", color=edge_color, penwidth=penwidth, arrowsize='0.9')

with dot.subgraph(name='legend') as leg:
    leg.attr(rank='sink', rankdir='LR', fontsize='9', fontname='Times New Roman')

    leg.node('nf', 'Feature', **{**NODE_STYLES['feature'], 'width': '0.5', 'height': '0.3'})
    leg.node('nm', 'Mode', **{**NODE_STYLES['mode'], 'width': '0.5', 'height': '0.3'})

    for (low, _), style in CONFIDENCE_BINS.items():
        html = f'''<
          <table border="0" cellborder="0" cellpadding="0">
            <tr>
              <td width="12" height="12" bgcolor="{style['color']}"></td>
              <td width="4"></td>
              <td>{style['label']}</td>
            </tr>
          </table>
        >'''
        leg.node(f'leg{low}', html, shape='plaintext')

    leg.node('wt', 'Width ∝ confidence', shape='plaintext')

    seq = ['nf', 'nm'] + [f'leg{low}' for low, _ in CONFIDENCE_BINS] + ['wt']
    for a, b in zip(seq, seq[1:]):
        leg.edge(a, b, style='invis')

dot.render('Loan_LostLinking_Network', view=False, cleanup=True)
print("Loan_LostLinking_Network.png")