async function fetchGraphData() {
    const errorBanner = document.getElementById('error-banner');
    const statusText = document.getElementById('status-text');
    
    errorBanner.classList.add('hidden');
    statusText.textContent = 'Fetching graph data...';

    try {
        // Fetch from the API endpoint with a default limit
        const response = await fetch('/api/graph/nodes-and-edges?limit=100');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (e) {
        console.error('Fetch error:', e);
        errorBanner.textContent = `Error loading graph: ${e.message}`;
        errorBanner.classList.remove('hidden');
        statusText.textContent = 'Failed to load graph.';
        return null;
    }
}

function initCytoscape(elements) {
    return cytoscape({
        container: document.getElementById('cy'),
        elements: elements,
        style: [
            {
                selector: 'node',
                style: {
                    'background-color': '#666',
                    'label': 'data(label)',
                    'color': '#333',
                    'font-size': '12px',
                    'text-valign': 'center',
                    'text-halign': 'right',
                    'text-margin': '5px'
                }
            },
            {
                selector: 'edge',
                style: {
                    'width': 2,
                    'line-color': '#ccc',
                    'target-arrow-color': '#ccc',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'label': 'data(label)',
                    'font-size': '10px',
                    'text-rotation': 'auto',
                    'text-margin-y': '-10px'
                }
            }
        ],
        layout: {
            name: 'cose',
            nodeRepulsion: 4000,
            idealEdgeLength: 100
        }
    });
}

let cy = null;

async function loadGraph() {
    const data = await fetchGraphData();
    if (!data) return;

    const elements = [...data.nodes, ...data.edges];
    
    if (cy) {
        cy.destroy();
    }
    
    cy = initCytoscape(elements);
    document.getElementById('status-text').textContent = `Graph loaded: ${data.nodes.length} nodes, ${data.edges.length} edges`;
}

document.getElementById('refresh-btn').addEventListener('click', loadGraph);

// Initial load
window.addEventListener('DOMContentLoaded', loadGraph);
