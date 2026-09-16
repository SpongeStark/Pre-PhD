// Configuration
const API_BASE = ""; // Relative to server
let activePolls = {};
let systemState = {};

// On Load
document.addEventListener("DOMContentLoaded", () => {
    // Initial fetch of system status
    updateSystemStatus();
    // Load initial forecast metrics
    loadForecastMetrics();
    // Start continuous status updates every 2 seconds
    setInterval(updateSystemStatus, 2000);
});

// Switch Dashboard Tabs
function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.remove("active"));
    
    // Find tab button that matches tabId
    event.target.classList.add("active");
    document.getElementById(`tab-${tabId}`).classList.add("active");
}

// Switch Visualizations Gallery Tabs
function switchGallery(plotId) {
    document.querySelectorAll(".gallery-tab").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".gallery-img-container").forEach(c => c.classList.add("hidden"));
    
    event.target.classList.add("active");
    document.getElementById(`gallery-${plotId}`).classList.remove("hidden");
}

// Toggle Individual Card Terminal Logs
function toggleConsole(scriptKey) {
    const consoleBox = document.getElementById(`log-${scriptKey}`);
    const header = consoleBox.previousElementSibling;
    
    if (consoleBox.classList.contains("collapsed")) {
        consoleBox.classList.remove("collapsed");
        header.classList.remove("collapsed");
    } else {
        consoleBox.classList.add("collapsed");
        header.classList.add("collapsed");
    }
}

// API Call: Update System Status
async function updateSystemStatus() {
    try {
        const res = await fetch(`${API_BASE}/api/status`);
        if (!res.ok) throw new Error("Server communication error");
        
        const data = await res.json();
        const previousState = JSON.parse(JSON.stringify(systemState));
        systemState = data;
        
        // Update connection status badge
        const indicator = document.getElementById("status-indicator");
        const statusText = document.getElementById("status-text");
        
        // Check if any script is running
        const anyRunning = Object.keys(data).some(key => data[key].status === "running");
        
        if (anyRunning) {
            indicator.className = "status-indicator busy";
            statusText.textContent = "SYSTEM BUSY";
        } else {
            indicator.className = "status-indicator online";
            statusText.textContent = "SYSTEM ONLINE";
        }
        
        // Update cards and pipeline nodes
        updateCards(data, previousState);
        updatePipelineNodes(data);
        
    } catch (err) {
        console.error("Status Update Failed: ", err);
        const indicator = document.getElementById("status-indicator");
        const statusText = document.getElementById("status-text");
        indicator.className = "status-indicator offline";
        statusText.textContent = "DISCONNECTED";
    }
}

// Update DOM cards based on script states
function updateCards(state, previousState = {}) {
    // 1. Standard single modules (PV, Con)
    ["pv_cleaner", "con_cleaner"].forEach(key => {
        const card = document.getElementById(`card-${key}`);
        if (!card) return;
        
        const statusBadge = card.querySelector(".status-badge");
        const runBtn = card.querySelector(".btn-primary");
        const currentStatus = state[key]?.status || "idle";
        const previousStatus = previousState[key]?.status || null;
        
        statusBadge.className = `status-badge ${currentStatus}`;
        statusBadge.textContent = currentStatus;
        
        if (currentStatus === "running") {
            card.classList.add("running");
            card.classList.remove("success", "error");
            runBtn.disabled = true;
            runBtn.textContent = "Executing...";
            startLogPolling(key);
        } else {
            card.classList.remove("running");
            runBtn.disabled = false;
            runBtn.textContent = `Run ${getNiceName(key)}`;
            
            if (currentStatus === "success") {
                card.classList.add("success");
                card.classList.remove("error");
                stopLogPolling(key);
                if (previousStatus !== "success") {
                    displayPlot(key);
                }
            } else if (currentStatus === "error") {
                card.classList.add("error");
                card.classList.remove("success");
                stopLogPolling(key);
            }
        }
    });

    // 2. Battery Optimizer (Triple Mode: Supermarket Base, Lidl EV, & Caltech EV)
    const cardOpt = document.getElementById("card-battery_optimizer");
    if (cardOpt) {
        const btnBase = document.getElementById("btn-run-base");
        const btnEV = document.getElementById("btn-run-ev");
        const btnCaltech = document.getElementById("btn-run-caltech");
        const statusBadgeOpt = cardOpt.querySelector(".status-badge");
        const modeBadge = document.getElementById("card-metric-mode");
        const resultsPanel = document.getElementById("results-battery_optimizer");
        
        const optBase = state["battery_optimizer"] || { status: "idle", logs: "" };
        const optEV = state["battery_optimizer_ev"] || { status: "idle", logs: "" };
        const optCaltech = state["battery_optimizer_caltech"] || { status: "idle", logs: "" };
        const prevBase = previousState["battery_optimizer"] || { status: "idle" };
        const prevEV = previousState["battery_optimizer_ev"] || { status: "idle" };
        const prevCaltech = previousState["battery_optimizer_caltech"] || { status: "idle" };
        
        const isBaseRunning = optBase.status === "running";
        const isEVRunning = optEV.status === "running";
        const isCaltechRunning = optCaltech.status === "running";
        
        if (isBaseRunning || isEVRunning || isCaltechRunning) {
            cardOpt.classList.add("running");
            cardOpt.classList.remove("success", "error");
            statusBadgeOpt.className = "status-badge running";
            
            if (isBaseRunning) {
                statusBadgeOpt.textContent = "running (supermarket)";
                if (btnBase) { btnBase.disabled = true; btnBase.textContent = "Executing Base..."; }
                if (btnEV) { btnEV.disabled = true; }
                if (btnCaltech) { btnCaltech.disabled = true; }
                startLogPolling("battery_optimizer");
            } else if (isEVRunning) {
                statusBadgeOpt.textContent = "running (lidl ev)";
                if (btnEV) { btnEV.disabled = true; btnEV.textContent = "Executing Lidl EV..."; }
                if (btnBase) { btnBase.disabled = true; }
                if (btnCaltech) { btnCaltech.disabled = true; }
                startLogPolling("battery_optimizer_ev");
            } else {
                statusBadgeOpt.textContent = "running (caltech ev)";
                if (btnCaltech) { btnCaltech.disabled = true; btnCaltech.textContent = "Executing Caltech EV..."; }
                if (btnBase) { btnBase.disabled = true; }
                if (btnEV) { btnEV.disabled = true; }
                startLogPolling("battery_optimizer_caltech");
            }
            if (resultsPanel) resultsPanel.classList.add("hidden");
        } else {
            cardOpt.classList.remove("running");
            if (btnBase) { btnBase.disabled = false; btnBase.textContent = "Run Supermarket Only"; }
            if (btnEV) { btnEV.disabled = false; btnEV.textContent = "⚡ Run EV Chargers Only"; }
            if (btnCaltech) { btnCaltech.disabled = false; btnCaltech.textContent = "🎓 Run Caltech EV Only"; }
            
            // Check if Caltech EV just succeeded
            if (optCaltech.status === "success" && (prevCaltech.status !== "success" || !resultsPanel || resultsPanel.classList.contains("hidden") && modeBadge?.classList.contains("caltech"))) {
                cardOpt.classList.add("success");
                cardOpt.classList.remove("error");
                statusBadgeOpt.className = "status-badge success";
                statusBadgeOpt.textContent = "success (caltech)";
                stopLogPolling("battery_optimizer_caltech");
                
                if (modeBadge) {
                    modeBadge.textContent = "🎓 Caltech EV Only";
                    modeBadge.className = "mode-badge caltech";
                }
                extractAndDisplaySizingMetrics(optCaltech.logs, "card-metric");
                if (resultsPanel) resultsPanel.classList.remove("hidden");
                displayPlot("battery_optimizer_caltech");
            }
            // Check if EV just succeeded
            else if (optEV.status === "success" && (prevEV.status !== "success" || !resultsPanel || resultsPanel.classList.contains("hidden") && modeBadge?.classList.contains("ev"))) {
                cardOpt.classList.add("success");
                cardOpt.classList.remove("error");
                statusBadgeOpt.className = "status-badge success";
                statusBadgeOpt.textContent = "success (lidl ev)";
                stopLogPolling("battery_optimizer_ev");
                
                if (modeBadge) {
                    modeBadge.textContent = "⚡ EV Chargers Only";
                    modeBadge.className = "mode-badge ev";
                }
                extractAndDisplaySizingMetrics(optEV.logs, "card-metric");
                if (resultsPanel) resultsPanel.classList.remove("hidden");
                displayPlot("battery_optimizer_ev");
            }
            // Check if Base just succeeded
            else if (optBase.status === "success" && (prevBase.status !== "success" || !resultsPanel || resultsPanel.classList.contains("hidden") && !modeBadge?.classList.contains("ev") && !modeBadge?.classList.contains("caltech"))) {
                cardOpt.classList.add("success");
                cardOpt.classList.remove("error");
                statusBadgeOpt.className = "status-badge success";
                statusBadgeOpt.textContent = "success (supermarket)";
                stopLogPolling("battery_optimizer");
                
                if (modeBadge) {
                    modeBadge.textContent = "Supermarket Only";
                    modeBadge.className = "mode-badge";
                }
                extractAndDisplaySizingMetrics(optBase.logs, "card-metric");
                if (resultsPanel) resultsPanel.classList.remove("hidden");
                displayPlot("battery_optimizer");
            }
            // Error handling
            else if (optBase.status === "error" || optEV.status === "error" || optCaltech.status === "error") {
                cardOpt.classList.add("error");
                cardOpt.classList.remove("success");
                statusBadgeOpt.className = "status-badge error";
                statusBadgeOpt.textContent = "error";
                stopLogPolling("battery_optimizer");
                stopLogPolling("battery_optimizer_ev");
                stopLogPolling("battery_optimizer_caltech");
            }
        }
    }

    // Pipeline Specific Updates
    const pState = state["pipeline"];
    const prevPState = previousState["pipeline"];
    const prevPStatus = prevPState ? prevPState.status : null;
    const pBtn = document.getElementById("btn-run-pipeline");
    const pStatusText = document.getElementById("pipeline-status-text");
    const pFill = document.getElementById("pipeline-fill");
    const pPercent = document.getElementById("pipeline-percentage");
    
    pStatusText.textContent = pState.status.toUpperCase();
    
    if (pState.status === "running") {
        pBtn.disabled = true;
        pBtn.textContent = "Running Pipeline...";
        startLogPolling("pipeline");
        
        // Calculate progress percentage based on current step
        const stepsOrder = ["pv_cleaner", "con_cleaner", "battery_optimizer"];
        const stepIdx = stepsOrder.indexOf(pState.current_step);
        const percent = stepIdx !== -1 ? Math.round((stepIdx / stepsOrder.length) * 100) : 10;
        pFill.style.width = `${percent}%`;
        pPercent.textContent = `${percent}%`;
    } else {
        pBtn.disabled = false;
        pBtn.textContent = "Run Sizing Pipeline";
        stopLogPolling("pipeline");
        
        if (pState.status === "success") {
            pFill.style.width = "100%";
            pPercent.textContent = "100%";
            
            // Only update metrics and gallery plots when transitioning to success
            if (prevPStatus !== "success") {
                extractAndDisplaySizingMetrics(pState.logs, "metric");
                // Also update the card metrics for consistency
                extractAndDisplaySizingMetrics(pState.logs, "card-metric");
                const resultsPanel = document.getElementById("results-battery_optimizer");
                if (resultsPanel) resultsPanel.classList.remove("hidden");
                // Load all gallery images
                displayAllGalleryPlots();
            }
        } else if (pState.status === "error") {
            pFill.style.width = "100%";
            pFill.style.background = "var(--neon-red)";
            pPercent.textContent = "FAILED";
        } else {
            pFill.style.width = "0%";
            pPercent.textContent = "0%";
        }
    }
}

// Update the glowing Pipeline Nodes at the top
function updatePipelineNodes(state) {
    const nodes = ["pv_cleaner", "con_cleaner", "battery_optimizer"];
    
    // Check if pipeline is running
    const pipelineRunning = state["pipeline"].status === "running";
    const pipelineCurrentStep = state["pipeline"].current_step;
    
    nodes.forEach(key => {
        const node = document.getElementById(`node-${key}`);
        if (!node) return;
        
        node.className = "step-node"; // reset
        
        if (pipelineRunning) {
            if (pipelineCurrentStep === key) {
                node.classList.add("active");
            } else {
                const stepsOrder = ["pv_cleaner", "con_cleaner", "battery_optimizer"];
                if (stepsOrder.indexOf(key) < stepsOrder.indexOf(pipelineCurrentStep)) {
                    node.classList.add("success");
                }
            }
        } else {
            // Match individual execution states
            node.classList.add(state[key].status);
        }
    });
}

// Log Polling Mechanism
function startLogPolling(scriptKey) {
    if (activePolls[scriptKey]) return; // already polling
    
    let targetBoxId = `log-${scriptKey}`;
    if (scriptKey === "battery_optimizer" || scriptKey === "battery_optimizer_ev" || scriptKey === "battery_optimizer_caltech") {
        targetBoxId = "log-battery_optimizer";
    } else if (scriptKey.startsWith("forecast_")) {
        targetBoxId = "log-forecast";
    }
    
    const pollFunc = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/logs?script=${scriptKey}`);
            if (!res.ok) throw new Error("Log fetch error");
            const data = await res.json();
            
            const logBox = document.getElementById(targetBoxId);
            if (logBox) {
                const pre = logBox.querySelector("pre");
                pre.textContent = data.logs;
                // Auto scroll to bottom
                logBox.scrollTop = logBox.scrollHeight;
            }
            
            // If the script finished, stop polling
            if (data.status !== "running") {
                stopLogPolling(scriptKey);
                updateSystemStatus(); // refresh status immediately
                if (scriptKey.startsWith("forecast_")) {
                    loadForecastMetrics();
                }
            }
        } catch (err) {
            console.error("Log Poll Failed:", err);
            stopLogPolling(scriptKey);
        }
    };
    
    // Run immediately and then every 800ms
    pollFunc();
    activePolls[scriptKey] = setInterval(pollFunc, 800);
}

function stopLogPolling(scriptKey) {
    if (activePolls[scriptKey]) {
        clearInterval(activePolls[scriptKey]);
        delete activePolls[scriptKey];
    }
}

// Trigger script execution
async function runScript(scriptKey) {
    try {
        const res = await fetch(`${API_BASE}/api/run`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ script: scriptKey })
        });
        
        if (!res.ok) {
            const errData = await res.json();
            alert(errData.error || "Failed to start execution");
            return;
        }
        
        let targetBoxId = `log-${scriptKey}`;
        if (scriptKey === "battery_optimizer" || scriptKey === "battery_optimizer_ev" || scriptKey === "battery_optimizer_caltech") {
            targetBoxId = "log-battery_optimizer";
        } else if (scriptKey.startsWith("forecast_")) {
            targetBoxId = "log-forecast";
        }
            
        // Collapse other logs and expand the current one
        document.querySelectorAll(".console-log").forEach(box => {
            if (box.id !== targetBoxId && box.id !== "log-pipeline" && box.id !== "log-forecast") {
                box.classList.add("collapsed");
                if (box.previousElementSibling) box.previousElementSibling.classList.add("collapsed");
            }
        });
        
        const currentBox = document.getElementById(targetBoxId);
        if (currentBox) {
            currentBox.classList.remove("collapsed");
            if (currentBox.previousElementSibling) currentBox.previousElementSibling.classList.remove("collapsed");
        }
        
        updateSystemStatus();
    } catch (err) {
        alert("Server error: " + err.message);
    }
}

// Trigger complete pipeline execution
async function runPipeline() {
    try {
        const res = await fetch(`${API_BASE}/api/run`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ script: "pipeline" })
        });
        
        if (!res.ok) {
            const errData = await res.json();
            alert(errData.error || "Failed to start pipeline");
            return;
        }
        
        updateSystemStatus();
    } catch (err) {
        alert("Server error: " + err.message);
    }
}

// Helper: map script key to plot image name
function getPlotFileName(scriptKey) {
    const mapping = {
        "pv_cleaner": "data_cleaner_results.png",
        "con_cleaner": "data_cleaner_con_results.png",
        "battery_optimizer": "optimization_results.png",
        "battery_optimizer_ev": "optimization_results_ev.png",
        "battery_optimizer_caltech": "optimization_results_caltech.png"
    };
    return mapping[scriptKey];
}

// Load and display a plot in individual cards
function displayPlot(scriptKey) {
    const filename = getPlotFileName(scriptKey);
    if (!filename) return;
    
    const containerId = (scriptKey === "battery_optimizer" || scriptKey === "battery_optimizer_ev" || scriptKey === "battery_optimizer_caltech") 
        ? "viz-battery_optimizer" 
        : `viz-${scriptKey}`;
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Add cache-busting timestamp
    const t = new Date().getTime();
    container.innerHTML = `<img src="${API_BASE}/api/plots/${filename}?t=${t}" alt="Plot Result" class="viz-img" onerror="imgError(this)">`;
}

// Load all plots into the gallery tabs
function displayAllGalleryPlots() {
    const t = new Date().getTime();
    
    const pv = document.querySelector("#gallery-pv img");
    if (pv) pv.src = `${API_BASE}/api/plots/data_cleaner_results.png?t=${t}`;
    
    const con = document.querySelector("#gallery-con img");
    if (con) con.src = `${API_BASE}/api/plots/data_cleaner_con_results.png?t=${t}`;
    
    const bat = document.querySelector("#gallery-battery img");
    if (bat) bat.src = `${API_BASE}/api/plots/optimization_results.png?t=${t}`;
    
    const batEV = document.querySelector("#gallery-battery_ev img");
    if (batEV) batEV.src = `${API_BASE}/api/plots/optimization_results_ev.png?t=${t}`;
    
    const batCaltech = document.querySelector("#gallery-battery_caltech img");
    if (batCaltech) batCaltech.src = `${API_BASE}/api/plots/optimization_results_caltech.png?t=${t}`;
}

// Image load fail fallback
function imgError(img) {
    const container = img.parentElement;
    img.style.display = "none";
    const placeholder = container.querySelector(".img-placeholder") || container.querySelector(".viz-placeholder");
    if (placeholder) {
        placeholder.style.display = "flex";
        placeholder.textContent = "Plot generated, but could not be loaded. Please run the script again.";
    }
}

// Parse battery sizing metrics from logs using regex
function extractAndDisplaySizingMetrics(logs, prefix = "metric") {
    const capacityMatch = logs.match(/Battery Capacity \(E_B_max\):\s*([\d\.]+)\s*kWh/i);
    const powerMatch = logs.match(/Battery Rated Power \(P_B_max\):\s*([\d\.]+)\s*kW/i);
    const costMatch = logs.match(/Total Annualized Cost \(CAPEX \+ OPEX\):\s*[^0-9\r\n]*([\d\.,]+)/i);
    
    const capEl = document.getElementById(`${prefix}-capacity`);
    const powEl = document.getElementById(`${prefix}-power`);
    const costEl = document.getElementById(`${prefix}-cost`);
    
    if (capEl) {
        if (capacityMatch && capacityMatch[1]) {
            capEl.textContent = `${parseFloat(capacityMatch[1]).toFixed(2)} kWh`;
        } else {
            capEl.textContent = "-- kWh";
        }
    }
    
    if (powEl) {
        if (powerMatch && powerMatch[1]) {
            powEl.textContent = `${parseFloat(powerMatch[1]).toFixed(2)} kW`;
        } else {
            powEl.textContent = "-- kW";
        }
    }
    
    if (costEl) {
        if (costMatch && costMatch[1]) {
            costEl.textContent = `€${parseFloat(costMatch[1].replace(/,/g, '')).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        } else {
            costEl.textContent = "-- €";
        }
    }
}

// Copy Log text helper
function copyLogs(scriptKey) {
    const logBox = document.getElementById(`log-${scriptKey}`);
    if (!logBox) return;
    
    const text = logBox.querySelector("pre").textContent;
    navigator.clipboard.writeText(text).then(() => {
        const btn = event.target;
        const originalText = btn.textContent;
        btn.textContent = "Copied!";
        setTimeout(() => btn.textContent = originalText, 1500);
    }).catch(err => {
        console.error("Copy failed: ", err);
    });
}

function getNiceName(key) {
    const mapping = {
        "pv_cleaner": "PV Cleaner",
        "con_cleaner": "Consumption Cleaner",
        "battery_optimizer": "Battery Optimizer",
        "forecast_pv": "Solar PV Forecaster",
        "forecast_con": "Supermarket Forecaster",
        "forecast_ev": "Lidl EV Forecaster",
        "forecast_caltech": "Caltech EV Forecaster",
        "forecast_all": "Multi-Target Forecaster Engine"
    };
    return mapping[key] || key;
}

// Switch Forecasting Visualizations Gallery Tabs
function switchForecastGallery(plotId) {
    document.querySelectorAll("#forecast-gallery-tabs .gallery-tab").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll("#tab-forecasting .gallery-img-container").forEach(c => c.classList.add("hidden"));
    
    if (event && event.target) {
        event.target.classList.add("active");
    }
    const targetEl = document.getElementById(`fgallery-${plotId}`);
    if (targetEl) targetEl.classList.remove("hidden");
}

// Load and populate Forecast Metrics Table
async function loadForecastMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/forecast/metrics`);
        if (!res.ok) return;
        const data = await res.json();
        
        const tbody = document.querySelector("#forecast-metrics-table tbody");
        if (!tbody) return;
        
        if (!data || Object.keys(data).length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color: var(--text-muted); padding: 20px;">No forecast metrics found. Run the forecasters to generate benchmark scorecard.</td></tr>`;
            return;
        }
        
        let html = "";
        for (const [key, item] of Object.entries(data)) {
            const bestM = item.best_model;
            const stats = item.models[bestM];
            const r2 = stats.r2;
            const wape = stats.wape_pct !== undefined ? `${stats.wape_pct.toFixed(1)}%` : "--";
            const biasVal = stats.energy_bias_pct;
            const biasText = biasVal !== undefined ? `${biasVal > 0 ? '+' : ''}${biasVal.toFixed(1)}%` : "--";
            let badgeClass = "badge-sparse";
            let badgeText = "Sparse / Intermittent";
            if (r2 >= 0.70) {
                badgeClass = "badge-high";
                badgeText = "High Predictability";
            } else if (r2 >= 0.30) {
                badgeClass = "badge-mod";
                badgeText = "Moderate Predictability";
            }
            
            html += `
                <tr>
                    <td><strong>${item.target_name}</strong></td>
                    <td><span style="font-family: monospace; font-weight: bold; color: var(--neon-cyan); background: rgba(6, 182, 212, 0.1); padding: 3px 8px; border-radius: 4px;">${bestM}</span></td>
                    <td style="font-weight: bold; color: ${r2 >= 0.7 ? '#10b981' : (r2 >= 0.3 ? '#f59e0b' : '#ef4444')};">${r2.toFixed(4)}</td>
                    <td>${stats.mae.toFixed(2)} ${item.unit}</td>
                    <td>${stats.rmse.toFixed(2)} ${item.unit}</td>
                    <td style="font-weight: bold; color: ${stats.wape_pct < 50 ? '#10b981' : (stats.wape_pct < 100 ? '#f59e0b' : '#ef4444')};">${wape}</td>
                    <td style="color: ${Math.abs(biasVal || 0) < 10 ? '#10b981' : '#f59e0b'}; font-weight: 600;">${biasText}</td>
                    <td><span class="badge-level ${badgeClass}">${badgeText}</span></td>
                </tr>
            `;
        }
        tbody.innerHTML = html;
        
        // Refresh forecast plot images
        refreshForecastPlots();
    } catch (err) {
        console.error("Failed to load forecast metrics:", err);
    }
}

function refreshForecastPlots() {
    const t = new Date().getTime();
    const scorecardImg = document.querySelector("#fgallery-scorecard img");
    if (scorecardImg) scorecardImg.src = `${API_BASE}/api/plots/forecast_scorecard.png?t=${t}`;
    
    const pvImg = document.querySelector("#fgallery-pv img");
    if (pvImg) pvImg.src = `${API_BASE}/api/plots/forecast_pv.png?t=${t}`;
    
    const conImg = document.querySelector("#fgallery-con img");
    if (conImg) conImg.src = `${API_BASE}/api/plots/forecast_con.png?t=${t}`;
    
    const evImg = document.querySelector("#fgallery-ev img");
    if (evImg) evImg.src = `${API_BASE}/api/plots/forecast_ev.png?t=${t}`;
    
    const caltechImg = document.querySelector("#fgallery-caltech img");
    if (caltechImg) caltechImg.src = `${API_BASE}/api/plots/forecast_caltech.png?t=${t}`;
}
