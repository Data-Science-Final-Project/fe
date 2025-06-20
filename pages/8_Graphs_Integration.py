import streamlit as st
import json
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components
import os

@st.cache_data
def load_graph_json(path):
    nodes, relationships = {}, []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data["type"] == "node":
                nodes[data["id"]] = data
            elif data["type"] == "relationship":
                relationships.append(data)
    return nodes, relationships


# Fix: load from the data folder at the project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
graph_path = os.path.join(project_root, "data", "merged_graph.json")
nodes_dict, relationships_list = load_graph_json(graph_path)

st.title("Relationship Viewer")
st.markdown(
    "Search for a Case and explore its related Laws, Judgments, and Cases.")
# Persistent state
if "last_case_number" not in st.session_state:
    st.session_state.last_case_number = ""
if "last_selected_types" not in st.session_state:
    st.session_state.last_selected_types = []

col1, col2 = st.columns([1, 3])
with col1:
    show_all = st.checkbox("Show All Related Types")
with col2:
    selected_types = []

    st.markdown("**Include related types:**")
    for label in ["Law", "Case", "Judgment"]:
        selected = (
            True if show_all
            else label in st.session_state.get("last_selected_types", ["Law", "Case"])
        )
        checked = st.checkbox(
            f"{label}",
            value=selected,
            disabled=show_all,
            key=f"type_{label}"
        )
        if checked:
            selected_types.append(label)


@st.cache_data
def get_all_case_numbers():
    return sorted([
        node["properties"]["number"]
        for node in nodes_dict.values()
        if "Case" in node["labels"] and "number" in node["properties"]
    ])

case_numbers = get_all_case_numbers()

case_number = st.selectbox(
    "Enter Case Number",
    options=case_numbers,
    index=case_numbers.index(st.session_state.last_case_number)
    if st.session_state.last_case_number in case_numbers else 0,
    placeholder="Start typing a case number..."
)


def find_case_node_by_number(number):
    for node in nodes_dict.values():
        if "Case" in node["labels"] and node["properties"].get("number") == number:
            return node
    return None


def get_connected_nodes_and_edges(center_node, allowed_labels):
    center_id = center_node["id"]
    center_number = center_node["properties"].get("number")
    connected_nodes = {}
    edges = []

    for rel in relationships_list:
        start_node = rel["start"]
        end_node = rel["end"]

        if start_node["properties"].get("number") == center_number:
            other = end_node
            direction = ("start", "end")
        elif end_node["properties"].get("number") == center_number:
            other = start_node
            direction = ("end", "start")
        else:
            continue

        other_label = other["labels"][0]
        if other_label in allowed_labels:
            connected_nodes[other["id"]] = nodes_dict.get(other["id"], other)
            edges.append(
                (center_node["id"], other["id"], rel.get("label", "RELATED")))

    return list(connected_nodes.values()), edges


def render_pyvis(center_node, related_nodes, edge_list):
    net = Network(height="690px", width="100%",
                  bgcolor="#1e1e1e", font_color="white")

    center_id = center_node["id"]
    center_label = center_node["properties"].get("number", "Unknown")
    net.add_node(center_id, label=f"מספר תיק מלא: {center_label}", color="red", font={"size": 14,
                                                                                      "color": "#ffffff",
                                                                                      "face": "segoe ui",
                                                                                      "background": "#333333",
                                                                                      "strokeWidth": 0})

    for node in related_nodes:
        node_id = node["id"]
        label_type = node["labels"][0]
        label_text = node["properties"].get("name") or node["properties"].get(
            "title") or node["properties"].get("number") or "Unknown"
        color = {"Law": "green", "Case": "orange",
                 "Judgment": "blue"}.get(label_type, "gray")
        ntype = {"Law": "חוק", "Case": "מספר תיק",
                 "Judgment": "מספר מקוצר"}.get(label_type, "gray")
        net.add_node(node_id, label=f"{ntype}: {label_text}", title=label_text, color=color, font={"size": 14,
                                                                                                   "color": "#ffffff",
                                                                                                   "face": "segoe ui",
                                                                                                   "background": "#333333",
                                                                                                   "strokeWidth": 0})

    for source, target, label in edge_list:
        net.add_edge(source, target, color="lightpink", arrows="to", label=label, font={"size": 14,
                                                                                        "color": "#ffffff",
                                                                                        "face": "segoe ui",
                                                                                        "background": "#333333",
                                                                                        "strokeWidth": 0})

    net.repulsion(node_distance=180, central_gravity=0.2,
                  spring_length=200, spring_strength=0.08)
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(temp_path.name)
    # Patch the saved HTML to remove white borders
    with open(temp_path.name, "r", encoding="utf-8") as f:
        html = f.read()

    # Inject custom styles to remove border and background box
    html = html.replace(
        "<body>",
        "<body style='margin:0;background-color:#1e1e1e;'>"
    )

    html = html.replace(
        "<div id=\"mynetwork\"",
        "<div id=\"mynetwork\" style=\"border:none;box-shadow:none;\""
    )

    # Save patched version
    with open(temp_path.name, "w", encoding="utf-8") as f:
        f.write(html)

    return temp_path.name


# Determine when to display
display_case_number = st.session_state.last_case_number
display_selected_types = st.session_state.last_selected_types

# if search:
#     st.session_state.last_case_number = case_number
#     st.session_state.last_selected_types = selected_types
#     display_case_number = case_number
#     display_selected_types = selected_types

# Always display graph if previous search exists
if case_number and selected_types:
    st.session_state.last_case_number = case_number
    st.session_state.last_selected_types = selected_types

    node = find_case_node_by_number(case_number)
    if not node:
        st.warning("⚠️ Case not found in the dataset.")
    else:
        related_nodes, edges = get_connected_nodes_and_edges(node, selected_types)
        html_path = render_pyvis(node, related_nodes, edges)
        with open(html_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=700, scrolling=True)
        with st.expander("📤 Export Graph"):
            st.markdown("""
                **To export the graph:**

                1. Right-click on the graph.
                2. Choose **'Save image as...'** or use your browser's screenshot tool.
            """)


