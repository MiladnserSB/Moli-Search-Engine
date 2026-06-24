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
    
    // Autocomplete Dropdown
    const autocompleteDropdown = document.getElementById('autocomplete-dropdown');
    
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

    // Handle personalized query click
    const personalizedQueryLink = document.getElementById('personalized-query-link');
    if (personalizedQueryLink) {
        personalizedQueryLink.addEventListener('click', (e) => {
            e.preventDefault();
            searchInput.value = personalizedQueryLink.textContent;
            refinementBanner.style.display = 'none';
            triggerSearch();
        });
    }

    // Handle query expansion click
    const expandedQueryText = document.getElementById('expanded-query-text');
    if (expandedQueryText) {
        expandedQueryText.style.cursor = 'pointer';
        expandedQueryText.addEventListener('click', () => {
            searchInput.value = expandedQueryText.textContent;
            refinementBanner.style.display = 'none';
            triggerSearch();
        });
    }

    // Event listener for search
    searchBtn.addEventListener('click', () => {
        autocompleteDropdown.style.display = 'none';
        triggerSearch();
    });
    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            autocompleteDropdown.style.display = 'none';
            triggerSearch();
        }
    });

    // Autocomplete input listener
    searchInput.addEventListener('input', async () => {
        const query = searchInput.value.trim();
        if (query.length < 2) {
            autocompleteDropdown.style.display = 'none';
            return;
        }

        try {
            const response = await fetch(`${GATEWAY_URL}/api/autocomplete`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prefix: query,
                    dataset: datasetSelect.value,
                    user_id: 'default_user'
                })
            });

            if (!response.ok) return;
            const data = await response.json();
            const suggestions = data.suggestions || [];

            if (suggestions.length === 0) {
                autocompleteDropdown.style.display = 'none';
                return;
            }

            autocompleteDropdown.innerHTML = '';
            suggestions.forEach(item => {
                const div = document.createElement('div');
                div.className = 'autocomplete-item';
                
                // Show clock for history, search glass for new terms
                const icon = item.is_history ? '🕒' : '🔍';
                div.innerHTML = `<span class="autocomplete-icon" style="margin-right: 8px; opacity: 0.6;">${icon}</span><span class="autocomplete-text">${item.text}</span>`;
                
                div.addEventListener('click', () => {
                    searchInput.value = item.text;
                    autocompleteDropdown.style.display = 'none';
                    triggerSearch();
                });
                autocompleteDropdown.appendChild(div);
            });
            autocompleteDropdown.style.display = 'block';
        } catch (err) {
            console.error('Error fetching autocomplete suggestions:', err);
        }
    });

    // Close autocomplete when clicking outside
    document.addEventListener('click', (e) => {
        if (e.target !== searchInput && e.target !== autocompleteDropdown) {
            autocompleteDropdown.style.display = 'none';
        }
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

        // Fetch spelling correction, expansion, and alternatives (always-on query refinement)
        fetchQueryRefinement(query);

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

    // Parallel fetch of detailed query refinement metrics
    async function fetchQueryRefinement(query) {
        try {
            const response = await fetch(`${GATEWAY_URL}/api/refine`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    query: query,
                    dataset: datasetSelect.value,
                    user_id: 'default_user'
                })
            });
            if (!response.ok) return;

            const data = await response.json();
            
            // 1. Spelling Correction (Did you mean?)
            const spellingWrapper = document.getElementById('spelling-correction-wrapper');
            if (data.corrected_query && data.corrected_query.toLowerCase() !== query.toLowerCase()) {
                suggestedQueryLink.textContent = data.corrected_query;
                spellingWrapper.style.display = 'flex';
            } else {
                spellingWrapper.style.display = 'none';
            }

            // 2. Query Expansion
            const expansionWrapper = document.getElementById('query-expansion-wrapper');
            const targetCompare = data.corrected_query || query;
            if (data.expanded_query && data.expanded_query.toLowerCase() !== targetCompare.toLowerCase()) {
                document.getElementById('expanded-query-text').textContent = data.expanded_query;
                expansionWrapper.style.display = 'flex';
            } else {
                expansionWrapper.style.display = 'none';
            }

            // 2b. Personalized Query Expansion (History-based)
            const personalizedWrapper = document.getElementById('personalized-expansion-wrapper');
            const personalizedLink = document.getElementById('personalized-query-link');
            const baseExpandedCompare = data.expanded_query || targetCompare;
            if (personalizedWrapper && personalizedLink) {
                if (data.personalized_query && data.personalized_query.toLowerCase() !== baseExpandedCompare.toLowerCase()) {
                    personalizedLink.textContent = data.personalized_query;
                    personalizedWrapper.style.display = 'flex';
                } else {
                    personalizedWrapper.style.display = 'none';
                }
            }

            // 3. Alternative Queries
            const alternativeWrapper = document.getElementById('alternative-queries-wrapper');
            const alternativeList = document.getElementById('alternative-queries-list');
            if (data.alternative_queries && data.alternative_queries.length > 0) {
                alternativeList.innerHTML = '';
                data.alternative_queries.forEach(alt => {
                    const btn = document.createElement('a');
                    btn.href = '#';
                    btn.className = 'refine-alternative-btn';
                    btn.textContent = alt;
                    btn.addEventListener('click', (e) => {
                        e.preventDefault();
                        searchInput.value = alt;
                        triggerSearch();
                    });
                    alternativeList.appendChild(btn);
                });
                alternativeWrapper.style.display = 'flex';
            } else {
                alternativeWrapper.style.display = 'none';
            }

            // Show banner if any item is visible
            const isPersonalizedVisible = personalizedWrapper && personalizedWrapper.style.display === 'flex';
            if (spellingWrapper.style.display === 'flex' || 
                expansionWrapper.style.display === 'flex' || 
                isPersonalizedVisible ||
                alternativeWrapper.style.display === 'flex') {
                refinementBanner.style.display = 'block';
            } else {
                refinementBanner.style.display = 'none';
            }
        } catch (err) {
            console.error('Error fetching query refinement:', err);
        }
    }

    // Render results in HTML
    function renderResults(data) {
        resultsCountMeta.textContent = `Found ${data.results.length} documents in ${data.time_taken_ms.toFixed(2)} ms`;

        if (data.results.length === 0) {
            searchResultsList.innerHTML = '<div class="no-results">No documents found matching your query.</div>';
            return;
        }

        searchResultsList.innerHTML = '';

        // Render Vector Fusion Info Banner if active
        if (data.personalized_fusion_info && data.personalized_fusion_info.historical_queries && data.personalized_fusion_info.historical_queries.length > 0) {
            const fusionAlert = document.createElement('div');
            fusionAlert.className = 'info-banner';
            fusionAlert.style.marginBottom = '15px';
            fusionAlert.style.padding = '12px 15px';
            fusionAlert.style.background = 'rgba(59, 130, 246, 0.1)';
            fusionAlert.style.border = '1px solid rgba(59, 130, 246, 0.2)';
            fusionAlert.style.borderRadius = '8px';
            fusionAlert.style.fontSize = '0.9em';
            fusionAlert.style.color = '#93c5fd';
            fusionAlert.style.display = 'flex';
            fusionAlert.style.alignItems = 'center';
            fusionAlert.style.gap = '10px';
            
            const historyList = data.personalized_fusion_info.historical_queries.join(', ');
            fusionAlert.innerHTML = `
                <span style="font-size: 1.2em;">🤖</span>
                <div>
                    <strong>Vector Fusion Active:</strong> Combined query embedding vector (70%) with user interest vector (30%) derived from recent search history: 
                    <span style="color: #60a5fa; font-style: italic; font-weight: 500;">"${historyList}"</span>
                </div>
            `;
            searchResultsList.appendChild(fusionAlert);
        }
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
                    label: `Evaluation Scores on ${evalData.dataset.toUpperCase()}`,
                    data: [
                        evalData.map_score,
                        evalData.recall_score,
                        evalData.precision_at_k,
                        evalData.ndcg_score
                    ],
                    backgroundColor: [
                        'rgba(139, 92, 246, 0.85)', // purple
                        'rgba(59, 130, 246, 0.85)', // blue
                        'rgba(236, 72, 153, 0.85)', // pink
                        'rgba(16, 185, 129, 0.85)'  // green (solid emerald)
                    ],
                    borderColor: [
                        '#8b5cf6',
                        '#3b82f6',
                        '#ec4899',
                        '#10b981'
                    ],
                    borderWidth: 1.5,
                    borderRadius: 6, // rounded corners for bars
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#1f2937', // dark charcoal for readability
                            font: {
                                family: "'Outfit', 'Inter', sans-serif",
                                size: 12,
                                weight: '500'
                            },
                            padding: 15
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1f2937',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        cornerRadius: 6,
                        padding: 10
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1.0,
                        grid: {
                            color: 'rgba(0, 0, 0, 0.05)' // very subtle dark lines
                        },
                        ticks: {
                            color: '#4b5563', // readable charcoal grey
                            font: {
                                family: "'Inter', sans-serif",
                                size: 11
                            }
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#4b5563', // readable charcoal grey
                            font: {
                                family: "'Inter', sans-serif",
                                size: 11,
                                weight: '500'
                            }
                        }
                    }
                }
            }
        });
    }
});
