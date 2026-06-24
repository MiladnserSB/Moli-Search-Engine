// ==============================================================================
// FRONTEND CONTROLLER - WEB DASHBOARD
// ==============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const searchInput = document.getElementById('search-input');
    const searchBtn = document.getElementById('search-btn');
    const searchResultsList = document.getElementById('search-results-list');
    const resultsCountMeta = document.getElementById('results-count-meta');
    
    const datasetSelect = document.getElementById('dataset-select');
    const modelSelect = document.getElementById('model-select');
    const additionalFeaturesChk = document.getElementById('additional-features-chk');
    const advancedOptions = document.getElementById('advanced-options');
    
    // Sliders
    const k1Slider = document.getElementById('bm25-k1');
    const bSlider = document.getElementById('bm25-b');
    const k1Val = document.getElementById('k1-val');
    const bVal = document.getElementById('b-val');
    const bm25Params = document.getElementById('bm25-parameters');
    
    // Hybrid Sliders
    const alphaSlider = document.getElementById('hybrid-alpha');
    const alphaVal = document.getElementById('alpha-val');
    const hybridParams = document.getElementById('hybrid-parameters');
    
    // Evaluation Elements
    const runEvaluationBtn = document.getElementById('run-evaluation-btn');
    const evaluationDash = document.getElementById('evaluation-dash');
    const closeEvalBtn = document.getElementById('close-eval-btn');
    const valMap = document.getElementById('val-map');
    const valRecall = document.getElementById('val-recall');
    const valP10 = document.getElementById('val-p10');
    const valNdcg = document.getElementById('val-ndcg');
    
    // Refinement Link
    const refinementBanner = document.getElementById('query-refinement-banner');
    const suggestedQueryLink = document.getElementById('suggested-query-link');

    // Clustering
    const triggerClusteringBtn = document.getElementById('trigger-clustering-btn');
    const clustersCountInput = document.getElementById('clusters-count');
    const clusteringStatus = document.getElementById('clustering-status');
    
    let chartInstance = null;

    // Toggle slider visibility
    modelSelect.addEventListener('change', (e) => {
        const val = e.target.value;
        if (val === 'bm25' || val.includes('hybrid')) {
            bm25Params.style.display = 'block';
        } else {
            bm25Params.style.display = 'none';
        }
        
        if (val.includes('hybrid')) {
            hybridParams.style.display = 'block';
        } else {
            hybridParams.style.display = 'none';
        }
    });

    // Update slider values dynamically
    k1Slider.addEventListener('input', (e) => {
        k1Val.textContent = e.target.value;
    });
    bSlider.addEventListener('input', (e) => {
        bVal.textContent = e.target.value;
    });
    alphaSlider.addEventListener('input', (e) => {
        alphaVal.textContent = e.target.value;
    });

    // Toggle Advanced Options block
    additionalFeaturesChk.addEventListener('change', (e) => {
        if (e.target.checked) {
            advancedOptions.style.display = 'block';
        } else {
            advancedOptions.style.display = 'none';
        }
    });

    // Handle suggested query click
    suggestedQueryLink.addEventListener('click', (e) => {
        e.preventDefault();
        searchInput.value = suggestedQueryLink.textContent;
        refinementBanner.style.display = 'none';
        triggerSearch();
    });

    // Event listener for search
    searchBtn.addEventListener('click', triggerSearch);
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') triggerSearch();
    });

    // Perform Search Action
    async function triggerSearch() {
        const query = searchInput.value.trim();
        if (!query) return;

        resultsCountMeta.textContent = 'Searching...';
        searchResultsList.innerHTML = '<div class="loader">Loading results...</div>';
        refinementBanner.style.display = 'none';

        const requestData = {
            query: query,
            dataset: datasetSelect.value,
            method: modelSelect.value,
            bm25_k1: parseFloat(k1Slider.value),
            bm25_b: parseFloat(bSlider.value),
            hybrid_alpha: parseFloat(alphaSlider.value),
            use_additional_features: additionalFeaturesChk.checked,
            top_k: 10
        };

        try {
            const response = await fetch(`${GATEWAY_URL}/api/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) throw new Error('Failed to retrieve search results');

            const data = await response.json();
            renderResults(data);
        } catch (error) {
            resultsCountMeta.textContent = 'Error occurred.';
            searchResultsList.innerHTML = `<div class="error-msg">${error.message}</div>`;
        }
    }

    // Render results in HTML
    function renderResults(data) {
        // Did you mean? spelling corrections check
        if (data.refined_query && data.refined_query.toLowerCase() !== searchInput.value.trim().toLowerCase()) {
            suggestedQueryLink.textContent = data.refined_query;
            refinementBanner.style.display = 'block';
        } else {
            refinementBanner.style.display = 'none';
        }

        resultsCountMeta.textContent = `Found ${data.results.length} documents in ${data.time_taken_ms.toFixed(2)} ms`;

        if (data.results.length === 0) {
            searchResultsList.innerHTML = '<div class="no-results">No documents found matching your query.</div>';
            return;
        }

        searchResultsList.innerHTML = '';
        data.results.forEach(doc => {
            const card = document.createElement('div');
            card.className = 'result-card';
            card.innerHTML = `
                <div class="result-header">
                    <h4 class="result-title">${doc.title || `Document`}</h4>
                    <span class="result-score">Sim Score: ${doc.score.toFixed(4)}</span>
                </div>
                <div class="result-meta" style="font-size: 0.85em; color: var(--text-secondary); margin-bottom: 8px;">
                    <strong>ID: ${doc.id}</strong>
                </div>
                <p class="result-content">${doc.content}</p>
            `;
            searchResultsList.appendChild(card);
        });
    }

    // Handle Document Clustering
    triggerClusteringBtn.addEventListener('click', async () => {
        clusteringStatus.textContent = 'Starting clustering job...';
        clusteringStatus.style.color = 'var(--text-secondary)';

        try {
            const response = await fetch(`${GATEWAY_URL}/api/cluster`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    dataset: datasetSelect.value,
                    num_clusters: parseInt(clustersCountInput.value),
                    algorithm: 'kmeans'
                })
            });

            if (!response.ok) throw new Error('Clustering request failed');
            const data = await response.json();
            
            clusteringStatus.textContent = `Status: ${data.status}. ${data.message}`;
            clusteringStatus.style.color = '#34d399';
        } catch (error) {
            clusteringStatus.textContent = `Error: ${error.message}`;
            clusteringStatus.style.color = '#f87171';
        }
    });

    // Handle System Evaluation Drawer
    runEvaluationBtn.addEventListener('click', async () => {
        evaluationDash.style.display = 'flex';
        valMap.textContent = 'Calculating...';
        valRecall.textContent = '...';
        valP10.textContent = '...';
        valNdcg.textContent = '...';

        const requestData = {
            dataset: datasetSelect.value,
            method: modelSelect.value,
            use_additional_features: additionalFeaturesChk.checked,
            bm25_k1: parseFloat(document.getElementById('bm25-k1').value),
            bm25_b: parseFloat(document.getElementById('bm25-b').value),
            hybrid_alpha: parseFloat(document.getElementById('hybrid-alpha').value)
        };

        try {
            const response = await fetch(`${GATEWAY_URL}/api/evaluate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) throw new Error('Evaluation request failed');
            const data = await response.json();

            // Update DOM
            valMap.textContent = data.map_score.toFixed(3);
            valRecall.textContent = data.recall_score.toFixed(3);
            valP10.textContent = data.precision_at_k.toFixed(3);
            valNdcg.textContent = data.ndcg_score.toFixed(3);

            // Redraw Chart
            renderChart(data);
        } catch (error) {
            valMap.textContent = 'Error';
            valRecall.textContent = 'Error';
            console.error(error);
        }
    });

    closeEvalBtn.addEventListener('click', () => {
        evaluationDash.style.display = 'none';
    });

    // Chart.js helper
    function renderChart(evalData) {
        const ctx = document.getElementById('metricsChart').getContext('2d');
        
        if (chartInstance) {
            chartInstance.destroy();
        }

        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['MAP', 'Recall', 'Precision@10', 'nDCG'],
                datasets: [{
                    label: `Scores on ${evalData.dataset.toUpperCase()}`,
                    data: [
                        evalData.map_score,
                        evalData.recall_score,
                        evalData.precision_at_k,
                        evalData.ndcg_score
                    ],
                    backgroundColor: [
                        'rgba(139, 92, 246, 0.6)', // purple
                        'rgba(59, 130, 246, 0.6)', // blue
                        'rgba(236, 72, 153, 0.6)', // pink
                        'rgba(52, 211, 153, 0.6)'  // green
                    ],
                    borderColor: [
                        '#8b5cf6',
                        '#3b82f6',
                        '#ec4899',
                        '#34d399'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: '#f3f4f6'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#9ca3af'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#9ca3af'
                        }
                    }
                }
            }
        });
    }
});
