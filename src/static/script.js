// Configuration
const API_BASE_URL = window.location.origin + '/api';
const SAMPLE_IMAGE_URL = 'https://via.placeholder.com/200';

// DOM elements
const categorySelect = document.getElementById('category-select');
const genderSelect = document.getElementById('gender-select');
const limitInput = document.getElementById('limit-input');
const updateButton = document.getElementById('update-button');
const uploadForm = document.getElementById('upload-form');
const fileInput = document.getElementById('file-input');
const loadingDiv = document.getElementById('loading');
const plotContainer = document.getElementById('plot-container');
const productDetail = document.getElementById('product-detail');
const closeDetailButton = document.getElementById('close-detail');

// Product detail elements
const productImage = document.getElementById('product-image');
const productTitle = document.getElementById('product-title');
const productId = document.getElementById('product-id');
const productCategory = document.getElementById('product-category');
const productGender = document.getElementById('product-gender');

// Store visualization data
let visualizationData = null;

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Update button click
    updateButton.addEventListener('click', fetchAndVisualizeFiltered);
    
    // Upload form submit
    uploadForm.addEventListener('submit', handleFileUpload);
    
    // Close detail button
    closeDetailButton.addEventListener('click', () => {
        productDetail.classList.add('hidden');
    });
    
    // Initial visualization (with demo data if available)
    fetchAndVisualizeDemo();
});

// Fetch and visualize demo data
// Fetch and visualize demo data
async function fetchAndVisualizeDemo() {
    try {
        showLoading();
        console.log('Fetching demo data...');
        
        // Try to get demo data from the server
        const response = await fetch(`${API_BASE_URL}/demo`);
        console.log('Demo response status:', response.status);
        
        if (response.ok) {
            const data = await response.json();
            console.log('Demo data received:', data);
            visualizationData = data;
            createVisualization(data.points);
        } else {
            console.error('Failed to fetch demo data:', await response.text());
            // If no demo data, just show empty message
            plotContainer.innerHTML = '<div class="empty-message">' +
                '<p>No visualization data available.</p>' +
                '<p>Use the controls to generate a visualization.</p>' +
                '</div>';
        }
    } catch (error) {
        console.error('Error fetching demo data:', error);
        // Create a simple error message in the plot container
        plotContainer.innerHTML = '<div class="error-message">' +
            '<p>Error loading visualization data.</p>' +
            '<p>Error details: ' + error.message + '</p>' +
            '</div>';
    } finally {
        hideLoading();
    }
}

// Fetch and visualize filtered data
// Fetch and visualize filtered data
async function fetchAndVisualizeFiltered() {
    const category = categorySelect.value;
    const gender = genderSelect.value;
    const limit = limitInput.value;
    
    console.log('Fetching filtered data:', { category, gender, limit });
    
    // Validation
    if (!category) {
        alert('Please select a category');
        return;
    }
    
    try {
        showLoading();
        
        // Construct URL with query parameters
        const url = new URL(`${API_BASE_URL}/filter`);
        url.searchParams.append('category', category);
        if (gender !== 'all') {
            url.searchParams.append('gender', gender);
        }
        url.searchParams.append('limit', limit);
        
        console.log('Fetching from URL:', url.toString());
        
        // Fetch data
        const response = await fetch(url);
        console.log('Filter response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            console.error('Error response:', errorText);
            throw new Error(`HTTP error ${response.status}: ${errorText}`);
        }
        
        // Process and visualize data
        const data = await response.json();
        console.log('Filter data received:', data);
        visualizationData = data;
        createVisualization(data.points);
        
    } catch (error) {
        console.error('Error fetching visualization data:', error);
        alert('Failed to fetch visualization data: ' + error.message);
        // Show error in visualization container
        plotContainer.innerHTML = '<div class="error-message">' +
            '<p>Error loading visualization data.</p>' +
            '<p>Error details: ' + error.message + '</p>' +
            '</div>';
    } finally {
        hideLoading();
    }
}

// Handle file upload
async function handleFileUpload(event) {
    event.preventDefault();
    
    const file = fileInput.files[0];
    if (!file) {
        alert('Please select a file to upload');
        return;
    }
    
    // Check file type
    if (!file.name.endsWith('.csv') && !file.name.endsWith('.json')) {
        alert('Please upload a CSV or JSON file');
        return;
    }
    
    try {
        showLoading();
        
        const formData = new FormData();
        formData.append('file', file);
        
        // Add additional parameters
        formData.append('pca_components', '50');
        formData.append('final_components', '2');
        
        // Upload file
        const response = await fetch(`${API_BASE_URL}/visualize/file`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error ${response.status}`);
        }
        
        // Process and visualize data
        const data = await response.json();
        visualizationData = data;
        createVisualization(data.points);
        
    } catch (error) {
        console.error('Error processing file:', error);
        alert('Failed to process file. Please check the format and try again.');
    } finally {
        hideLoading();
    }
}

// Create interactive visualization
function createVisualization(points) {
    if (!points || points.length === 0) {
        plotContainer.innerHTML = '<div class="empty-message"><p>No data available for visualization</p></div>';
        return;
    }
    
    // Extract coordinates and metadata
    const x = points.map(p => p.x);
    const y = points.map(p => p.y);
    
    // Create text for hover tooltips
    const text = points.map(p => p.title || `Product ${p.product_id}`);
    
    // Create colors based on categories
    const categoryColors = {
        'clothing': '#1f77b4',
        'footwear': '#ff7f0e',
        'accessories': '#2ca02c'
    };
    
    const colors = points.map(p => categoryColors[p.category] || '#7f7f7f');
    
    // Prepare data for Plotly
    const trace = {
        x: x,
        y: y,
        mode: 'markers',
        type: 'scatter',
        text: text,
        marker: {
            size: 8,
            color: colors,
            opacity: 0.7
        },
        hoverinfo: 'text'
    };
    
    // Layout configuration
    const layout = {
        title: 'Product Embedding Visualization',
        hovermode: 'closest',
        margin: { l: 40, r: 40, b: 40, t: 60 },
        xaxis: {
            title: 'Dimension 1',
            zeroline: false
        },
        yaxis: {
            title: 'Dimension 2',
            zeroline: false
        }
    };
    
    // Plot configuration
    const config = {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ['lasso2d', 'select2d']
    };
    
    // Create plot
    Plotly.newPlot(plotContainer, [trace], layout, config);
    
    // Add click event
    plotContainer.on('plotly_click', function(data) {
        const pointIndex = data.points[0].pointIndex;
        const clickedPoint = points[pointIndex];
        showProductDetail(clickedPoint);
    });
}

// Show product detail panel
function showProductDetail(product) {
    // Set product details
    productTitle.textContent = product.title || 'Product';
    productId.textContent = product.product_id;
    productCategory.textContent = product.category || 'N/A';
    productGender.textContent = product.gender || 'N/A';
    
    // Set image or placeholder
    if (product.image_url) {
        productImage.src = product.image_url;
    } else {
        productImage.src = SAMPLE_IMAGE_URL;
    }
    
    // Show the detail panel
    productDetail.classList.remove('hidden');
}

// Show loading indicator
function showLoading() {
    loadingDiv.classList.remove('hidden');
}

// Hide loading indicator
function hideLoading() {
    loadingDiv.classList.add('hidden');
}