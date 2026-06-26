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

    // Evaluation mode (offline / online)
    const evalModeOfflineRadio  = document.getElementById('eval-mode-offline');
    const evalModeOnlineRadio   = document.getElementById('eval-mode-online');
    const onlineLimitWrapper    = document.getElementById('online-limit-wrapper');
    const onlineLimitInput      = document.getElementById('online-limit-input');

    // Show / hide the query-limit input when switching mode
    function syncEvalModeUI() {
        const isOnline = evalModeOnlineRadio && evalModeOnlineRadio.checked;
        if (onlineLimitWrapper) {
            onlineLimitWrapper.style.display = isOnline ? 'block' : 'none';
        }
        if (runEvaluationBtn) {
            runEvaluationBtn.textContent = isOnline
                ? '🌐 Run Online Evaluation'
                : '⚡ Run Offline Evaluation';
        }
    }
    if (evalModeOfflineRadio) evalModeOfflineRadio.addEventListener('change', syncEvalModeUI);
    if (evalModeOnlineRadio)  evalModeOnlineRadio.addEventListener('change', syncEvalModeUI);
    syncEvalModeUI(); // initial state
    
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
    let clusteringChartInstance = null;

    // Tabs Navigation
    const tabSearch = document.getElementById('tab-search');
    const tabClustering = document.getElementById('tab-clustering');
    const searchResultsSection = document.getElementById('search-results-section');
    const clusteringSection = document.getElementById('clustering-section');
    const clusterCardsContainer = document.getElementById('cluster-cards-container');

    tabSearch.addEventListener('click', () => {
        tabSearch.classList.add('active');
        tabClustering.classList.remove('active');
        searchResultsSection.style.display = 'block';
        clusteringSection.style.display = 'none';
    });

    tabClustering.addEventListener('click', () => {
        tabClustering.classList.add('active');
        tabSearch.classList.remove('active');
        searchResultsSection.style.display = 'none';
        clusteringSection.style.display = 'block';
        fetchAndRenderClustering();
    });

    // Dismiss Refinement Banner
    const closeRefineBannerBtn = document.getElementById('close-refine-banner-btn');
    if (closeRefineBannerBtn) {
        closeRefineBannerBtn.addEventListener('click', () => {
            refinementBanner.style.display = 'none';
        });
    }

    // HSL color helpers for dynamic, aesthetic cluster visualization
    function getClusterColor(clusterId, totalClusters) {
        const hue = (clusterId * (360 / totalClusters)) % 360;
        return `hsla(${hue}, 70%, 55%, 0.85)`;
    }
    function getClusterBorderColor(clusterId, totalClusters) {
        const hue = (clusterId * (360 / totalClusters)) % 360;
        return `hsl(${hue}, 75%, 45%)`;
    }

    async function fetchAndRenderClustering() {
        const dataset = datasetSelect.value;
        const numClusters = parseInt(clustersCountInput.value) || 5;
        
        clusterCardsContainer.innerHTML = '<div class="loader" style="grid-column: 1/-1; text-align: center; padding: 2rem;">Loading clustering analysis...</div>';
        
        try {
            const response = await fetch(`${GATEWAY_URL}/api/cluster/plot/${dataset}/${numClusters}`);
            if (!response.ok) throw new Error('Failed to fetch clustering plot data');
            const data = await response.json();
            
            renderClusteringChart(data, numClusters);
            renderClusterCards(data, numClusters);
        } catch (error) {
            clusterCardsContainer.innerHTML = `<div class="error-msg" style="grid-column: 1/-1; text-align: center; padding: 2rem; color: #ef4444;">Error loading clusters: ${error.message}</div>`;
        }
    }

    function renderClusteringChart(data, numClusters) {
        const ctx = document.getElementById('clusteringChart').getContext('2d');
        if (clusteringChartInstance) {
            clusteringChartInstance.destroy();
        }

        // Group points by cluster
        const datasets = [];
        for (let c = 0; c < numClusters; c++) {
            const pointsInCluster = data.points.filter(p => p.cluster === c);
            datasets.push({
                label: data.cluster_names[c] || `Cluster ${c + 1}`,
                data: pointsInCluster.map(p => ({
                    x: p.x,
                    y: p.y,
                    doc_id: p.doc_id,
                    snippet: p.snippet
                })),
                backgroundColor: getClusterColor(c, numClusters),
                borderColor: getClusterBorderColor(c, numClusters),
                borderWidth: 1.5,
                pointRadius: 6,
                pointHoverRadius: 9,
                showLine: false
            });
        }

        // Add centroids dataset
        datasets.push({
            label: 'Cluster Centers (Centroids)',
            data: data.centroids.map(c => ({
                x: c.x,
                y: c.y,
                cluster: c.cluster
            })),
            backgroundColor: '#1f2937',
            borderColor: '#111827',
            borderWidth: 2,
            pointRadius: 11,
            pointHoverRadius: 13,
            pointStyle: 'crossRot', // renders as 'X' marker!
            showLine: false
        });

        clusteringChartInstance = new Chart(ctx, {
            type: 'scatter',
            data: { datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#1f2937',
                            font: {
                                family: "'Outfit', 'Inter', sans-serif",
                                size: 11,
                                weight: '500'
                            },
                            boxWidth: 12
                        }
                    },
                    tooltip: {
                        backgroundColor: '#1f2937',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        cornerRadius: 6,
                        padding: 10,
                        callbacks: {
                            label: function(context) {
                                const raw = context.raw;
                                if (raw.doc_id !== undefined) {
                                    return `ID: ${raw.doc_id} | ${raw.snippet}`;
                                }
                                return `Centroid (Cluster ${raw.cluster + 1})`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'PCA Component 1',
                            color: '#4b5563',
                            font: { family: "'Outfit', sans-serif", size: 12, weight: '600' }
                        },
                        grid: { color: 'rgba(0, 0, 0, 0.04)' },
                        ticks: { color: '#4b5563' }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'PCA Component 2',
                            color: '#4b5563',
                            font: { family: "'Outfit', sans-serif", size: 12, weight: '600' }
                        },
                        grid: { color: 'rgba(0, 0, 0, 0.04)' },
                        ticks: { color: '#4b5563' }
                    }
                }
            }
        });
    }

    function renderClusterCards(data, numClusters) {
        clusterCardsContainer.innerHTML = '';
        
        for (let c = 0; c < numClusters; c++) {
            const pointsInCluster = data.points.filter(p => p.cluster === c);
            const topDocs = pointsInCluster.slice(0, 3); // show top 3 docs
            
            const card = document.createElement('div');
            card.className = 'cluster-card';
            
            const headerColor = getClusterBorderColor(c, numClusters);
            
            let listItemsHtml = '';
            if (topDocs.length === 0) {
                listItemsHtml = '<li class="cluster-doc-item" style="color: var(--text-muted); font-style: italic;">No sampled documents in this cluster</li>';
            } else {
                topDocs.forEach(doc => {
                    listItemsHtml += `
                        <li class="cluster-doc-item">
                            <span class="cluster-doc-id">ID: ${doc.doc_id}</span>
                            <span class="cluster-doc-text">${doc.snippet}</span>
                        </li>
                    `;
                });
            }
            
            card.innerHTML = `
                <div class="cluster-card-header" style="background-color: ${headerColor};">
                    <span>${data.cluster_names[c] || `Cluster ${c+1}`}</span>
                </div>
                <div class="cluster-card-body">
                    <ul class="cluster-doc-list">
                        ${listItemsHtml}
                    </ul>
                </div>
            `;
            clusterCardsContainer.appendChild(card);
        }
    }

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

        // Ensure we switch to the search results tab when triggering a new search
        tabSearch.click();

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
            
            const spellingWrapper = document.getElementById('spelling-correction-wrapper');
            if (data.corrected_query && data.corrected_query.toLowerCase() !== query.toLowerCase()) {
                suggestedQueryLink.textContent = data.corrected_query;
                spellingWrapper.style.display = 'flex';
            } else {
                spellingWrapper.style.display = 'none';
            }

            const expansionWrapper = document.getElementById('query-expansion-wrapper');
            const targetCompare = data.corrected_query || query;
            if (data.expanded_query && data.expanded_query.toLowerCase() !== targetCompare.toLowerCase()) {
                document.getElementById('expanded-query-text').textContent = data.expanded_query;
                expansionWrapper.style.display = 'flex';
            } else {
                expansionWrapper.style.display = 'none';
            }

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

            // Automatically switch to tab and reload plot coordinates on demand
            tabClustering.click();
        } catch (error) {
            clusteringStatus.textContent = `Error: ${error.message}`;
            clusteringStatus.style.color = '#f87171';
        }
    });

    // Handle System Evaluation Drawer
    runEvaluationBtn.addEventListener('click', async () => {
        evaluationDash.style.display = 'flex';

        // Determine mode
        const isOnlineMode = evalModeOnlineRadio && evalModeOnlineRadio.checked;
        const evalMode = isOnlineMode ? 'online' : 'offline';
        const onlineLimit = onlineLimitInput ? parseInt(onlineLimitInput.value) || 500 : 500;
        
        // Base elements
        const baseMap    = document.getElementById('base-map');
        const baseRecall = document.getElementById('base-recall');
        const baseP10    = document.getElementById('base-p10');
        const baseNdcg   = document.getElementById('base-ndcg');
        
        // Improvement badges
        const improveMap    = document.getElementById('improve-map');
        const improveRecall = document.getElementById('improve-recall');
        const improveP10    = document.getElementById('improve-p10');
        const improveNdcg   = document.getElementById('improve-ndcg');

        // Labels for base/enhanced columns
        const baseModeEl     = document.getElementById('eval-base-mode');
        const enhancedModeEl = document.getElementById('eval-enhanced-mode');
        const evalStatusMsg  = document.getElementById('eval-status-msg');
        
        // ── Reset UI state ─────────────────────────────────────────────────
        [baseMap, baseRecall, baseP10, baseNdcg].forEach(el => el.textContent = '…');
        [valMap, valRecall, valP10, valNdcg].forEach(el => el.textContent = '…');
        [improveMap, improveRecall, improveP10, improveNdcg].forEach(el => {
            el.textContent = '';
            el.className = 'metric-improvement';
        });
        if (baseModeEl)     baseModeEl.textContent    = '';
        if (enhancedModeEl) enhancedModeEl.textContent = '';
        
        const modeLabel = isOnlineMode
            ? `⏳ Online mode: running Base evaluation (${onlineLimit} queries)…`
            : '⚡ Offline mode: loading pre-computed Base results from cache…';
        if (evalStatusMsg) evalStatusMsg.textContent = modeLabel;

        const requestData = {
            dataset:       datasetSelect.value,
            method:        modelSelect.value,
            bm25_k1:       parseFloat(document.getElementById('bm25-k1').value),
            bm25_b:        parseFloat(document.getElementById('bm25-b').value),
            hybrid_alpha:  parseFloat(document.getElementById('hybrid-alpha').value),
            mode:          evalMode,
            online_limit:  onlineLimit
        };

        try {
            // ── Step 1: Base evaluation ────────────────────────────────────
            const baseResponse = await fetch(`${GATEWAY_URL}/api/evaluate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...requestData, use_additional_features: false })
            });

            if (!baseResponse.ok) {
                const errText = await baseResponse.text();
                throw new Error(`Base evaluation failed (${baseResponse.status}): ${errText.slice(0, 200)}`);
            }
            const baseData = await baseResponse.json();

            // Update Base column
            baseMap.textContent    = baseData.map_score.toFixed(4);
            baseRecall.textContent = baseData.recall_score.toFixed(4);
            baseP10.textContent    = baseData.precision_at_k.toFixed(4);
            baseNdcg.textContent   = baseData.ndcg_score.toFixed(4);

            const baseQCount = baseData.num_queries_evaluated || '—';
            const baseMode   = baseData.mode || 'Base';
            if (baseModeEl) baseModeEl.textContent = `n=${baseQCount}`;

            // Update status for step 2
            const step2Label = isOnlineMode
                ? `✅ Base done (n=${baseQCount})! Running Enhanced evaluation (${onlineLimit} queries + spell correction)… may take 2–10 min for neural models.`
                : `✅ Base loaded (n=${baseQCount})! Loading Enhanced results from cache…`;
            if (evalStatusMsg) evalStatusMsg.textContent = step2Label;

            // ── Step 2: Enhanced evaluation ────────────────────────────────
            const enhancedResponse = await fetch(`${GATEWAY_URL}/api/evaluate`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ...requestData, use_additional_features: true })
            });

            if (!enhancedResponse.ok) {
                const errText = await enhancedResponse.text();
                throw new Error(`Enhanced evaluation failed (${enhancedResponse.status}): ${errText.slice(0, 200)}`);
            }
            const enhancedData = await enhancedResponse.json();

            // Update Enhanced column
            valMap.textContent    = enhancedData.map_score.toFixed(4);
            valRecall.textContent = enhancedData.recall_score.toFixed(4);
            valP10.textContent    = enhancedData.precision_at_k.toFixed(4);
            valNdcg.textContent   = enhancedData.ndcg_score.toFixed(4);

            const enhQCount = enhancedData.num_queries_evaluated || '—';
            const enhMode   = enhancedData.mode || 'Enhanced';
            if (enhancedModeEl) enhancedModeEl.textContent = `n=${enhQCount}`;

            if (evalStatusMsg) {
                evalStatusMsg.textContent =
                    `✅ Evaluation complete! Showing Before vs After Advanced Features. ` +
                    `Mode: ${evalMode.toUpperCase()} | Dataset: ${requestData.dataset} | Model: ${requestData.method.toUpperCase()}`;
            }

            // ── Improvement badges ─────────────────────────────────────────
            function updateImprovementBadge(badge, baseVal, enhancedVal) {
                const diff = enhancedVal - baseVal;
                let pct = 0.0;
                if (baseVal > 0) {
                    pct = (diff / baseVal) * 100;
                } else if (enhancedVal > 0) {
                    pct = 100.0;
                }
                const sign = diff > 0 ? '+' : '';
                if (diff > 0.0001) {
                    badge.textContent = `${sign}${pct.toFixed(1)}%`;
                    badge.classList.add('positive');
                } else if (diff < -0.0001) {
                    badge.textContent = `${pct.toFixed(1)}%`;
                    badge.classList.add('negative');
                } else {
                    badge.textContent = `≈ same`;
                    badge.classList.add('neutral');
                }
            }

            updateImprovementBadge(improveMap,    baseData.map_score,      enhancedData.map_score);
            updateImprovementBadge(improveRecall,  baseData.recall_score,   enhancedData.recall_score);
            updateImprovementBadge(improveP10,     baseData.precision_at_k, enhancedData.precision_at_k);
            updateImprovementBadge(improveNdcg,    baseData.ndcg_score,     enhancedData.ndcg_score);

            // Redraw grouped comparison Chart
            renderChart(baseData, enhancedData);

        } catch (error) {
            if (evalStatusMsg) evalStatusMsg.textContent = `❌ Error: ${error.message}`;
            valMap.textContent    = 'Error';
            valRecall.textContent = '–';
            valP10.textContent    = '–';
            valNdcg.textContent   = '–';
            console.error('[Evaluation Error]', error);
        }
    });

    closeEvalBtn.addEventListener('click', () => {
        evaluationDash.style.display = 'none';
    });

    // Chart.js helper for grouped comparison double-bar chart
    function renderChart(baseData, enhancedData) {
        const ctx = document.getElementById('metricsChart').getContext('2d');
        
        if (chartInstance) {
            chartInstance.destroy();
        }

        chartInstance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['MAP', 'Recall', 'Precision@10', 'nDCG'],
                datasets: [
                    {
                        label: 'Base System (Standard)',
                        data: [
                            baseData.map_score,
                            baseData.recall_score,
                            baseData.precision_at_k,
                            baseData.ndcg_score
                        ],
                        backgroundColor: 'rgba(148, 163, 184, 0.8)', // Slate grey (#94a3b8)
                        borderColor: '#94a3b8',
                        borderWidth: 1.5,
                        borderRadius: 6,
                        borderSkipped: false
                    },
                    {
                        label: 'Enhanced System (With Refinement & Fusion)',
                        data: [
                            enhancedData.map_score,
                            enhancedData.recall_score,
                            enhancedData.precision_at_k,
                            enhancedData.ndcg_score
                        ],
                        backgroundColor: 'rgba(0, 188, 242, 0.85)', // Sky blue (#00bcf2)
                        borderColor: '#00bcf2',
                        borderWidth: 1.5,
                        borderRadius: 6,
                        borderSkipped: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#1f2937',
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
                            color: 'rgba(0, 0, 0, 0.05)'
                        },
                        ticks: {
                            color: '#4b5563',
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
                            color: '#4b5563',
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
