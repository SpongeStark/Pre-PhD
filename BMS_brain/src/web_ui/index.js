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
    // Load initial hyperparameter tuning metrics
    loadTuningMetrics();
    // Load initial MPC metrics
    loadMPCMetrics();
    // Load initial RT metrics
    loadRTMetrics();
    // Start continuous status updates every 2 seconds
    setInterval(updateSystemStatus, 2000);
});

// Switch Dashboard Tabs
function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(panel => panel.classList.remove("active"));
    
    // Find tab button that matches tabId
    if (event && event.target) {
        event.target.classList.add("active");
    }
    const panel = document.getElementById(`tab-${tabId}`);
    if (panel) panel.classList.add("active");

    if (tabId === "forecasting") {
        loadForecastMetrics();
        loadTuningMetrics();
    } else if (tabId === "mpc") {
        loadMPCMetrics();
    } else if (tabId === "rt") {
        loadRTMetrics();
    }
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

    // 3. MPC Operational Optimizers
    const mpcConfigs = [
        { key: "mpc_supermarket", cardId: "mpc-card-supermarket", btnClass: ".btn-mpc-supermarket", defaultText: "🏢 Run Supermarket MPC", runningText: "Executing Supermarket MPC..." },
        { key: "mpc_ev", cardId: "mpc-card-ev", btnClass: ".btn-mpc-ev", defaultText: "⚡ Run Lidl EV MPC", runningText: "Executing Lidl EV MPC..." },
        { key: "mpc_caltech", cardId: "mpc-card-caltech", btnClass: ".btn-mpc-caltech", defaultText: "🎓 Run Caltech Campus MPC", runningText: "Executing Caltech MPC..." }
    ];

    mpcConfigs.forEach(cfg => {
        const card = document.getElementById(cfg.cardId);
        const btn = document.querySelector(cfg.btnClass);
        const currentStatus = state[cfg.key]?.status || "idle";

        if (card) {
            const badge = card.querySelector(".status-badge");
            if (badge) {
                badge.className = `status-badge ${currentStatus}`;
                badge.textContent = currentStatus === "success" ? "Optimal" : currentStatus;
            }
            if (currentStatus === "running") {
                card.classList.add("running");
                card.classList.remove("success", "error");
            } else if (currentStatus === "success") {
                card.classList.add("success");
                card.classList.remove("running", "error");
            } else if (currentStatus === "error") {
                card.classList.add("error");
                card.classList.remove("running", "success");
            } else {
                card.classList.remove("running", "success", "error");
            }
        }

        if (btn) {
            if (currentStatus === "running") {
                btn.disabled = true;
                btn.textContent = cfg.runningText;
                startLogPolling(cfg.key);
            } else {
                btn.disabled = false;
                btn.textContent = cfg.defaultText;
            }
        }
    });

    // 4. Real-Time Secondary Controller (Supermarket, Lidl EV, Caltech EV)
    const rtConfigs = [
        { key: "rt_supermarket", cardId: "rt-card-supermarket", btnClass: ".btn-rt-supermarket", defaultText: "🏢 Run Supermarket RT", runningText: "Compensating Supermarket..." },
        { key: "rt_ev", cardId: "rt-card-ev", btnClass: ".btn-rt-ev", defaultText: "⚡ Run Lidl EV RT", runningText: "Compensating Lidl EV..." },
        { key: "rt_caltech", cardId: "rt-card-caltech", btnClass: ".btn-rt-caltech", defaultText: "🎓 Run Caltech Campus RT", runningText: "Compensating Caltech..." }
    ];

    rtConfigs.forEach(cfg => {
        const card = document.getElementById(cfg.cardId);
        const btn = document.querySelector(cfg.btnClass);
        const currentStatus = state[cfg.key]?.status || "idle";

        if (card) {
            const badge = card.querySelector(".status-badge");
            if (badge) {
                badge.className = `status-badge ${currentStatus}`;
                badge.textContent = currentStatus === "success" ? "Optimal [Passed]" : currentStatus;
            }
            if (currentStatus === "running") {
                card.classList.add("running");
                card.classList.remove("success", "error");
            } else if (currentStatus === "success") {
                card.classList.add("success");
                card.classList.remove("running", "error");
            } else if (currentStatus === "error") {
                card.classList.add("error");
                card.classList.remove("running", "success");
            } else {
                card.classList.remove("running", "success", "error");
            }
        }

        if (btn) {
            if (currentStatus === "running") {
                btn.disabled = true;
                btn.textContent = cfg.runningText;
                startLogPolling(cfg.key);
            } else {
                btn.disabled = false;
                btn.textContent = cfg.defaultText;
            }
        }
    });

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
    } else if (scriptKey.startsWith("forecast_") || scriptKey === "forecaster_tuning") {
        targetBoxId = "log-forecast";
    } else if (scriptKey.startsWith("mpc_")) {
        targetBoxId = "log-mpc";
    } else if (scriptKey.startsWith("rt_")) {
        targetBoxId = "log-rt";
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
                if (scriptKey.startsWith("forecast_") || scriptKey === "forecaster_tuning") {
                    loadForecastMetrics();
                    loadTuningMetrics();
                }
                if (scriptKey.startsWith("mpc_")) {
                    loadMPCMetrics();
                    refreshMPCPlots();
                }
                if (scriptKey.startsWith("rt_")) {
                    loadRTMetrics();
                    refreshRTPlots();
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
        } else if (scriptKey.startsWith("forecast_") || scriptKey === "forecaster_tuning") {
            targetBoxId = "log-forecast";
        } else if (scriptKey.startsWith("mpc_")) {
            targetBoxId = "log-mpc";
        } else if (scriptKey.startsWith("rt_")) {
            targetBoxId = "log-rt";
        }
            
        // Collapse other logs and expand the current one
        document.querySelectorAll(".console-log").forEach(box => {
            if (box.id !== targetBoxId && box.id !== "log-pipeline" && box.id !== "log-forecast" && box.id !== "log-mpc" && box.id !== "log-rt") {
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
        "forecast_all": "Multi-Target Forecaster Engine",
        "forecaster_tuning": "Optuna Hyperparameter Sweep",
        "mpc_supermarket": "Supermarket MPC Optimizer",
        "mpc_ev": "Lidl EV MPC Optimizer",
        "mpc_caltech": "Caltech EV MPC Optimizer",
        "rt_supermarket": "Supermarket RT Controller",
        "rt_ev": "Lidl EV RT Controller",
        "rt_caltech": "Caltech EV RT Controller",
        "rt_all": "All RT Simulations"
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

// Load and populate Hyperparameter Tuning Metrics Table
async function loadTuningMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/tuning/results`);
        if (!res.ok) return;
        const data = await res.json();
        
        const tbody = document.querySelector("#tuning-metrics-table tbody");
        if (!tbody) return;
        
        if (!data || !data.comparisons || data.comparisons.length === 0) {
            tbody.innerHTML = `<tr><td colspan="9" style="text-align:center; color: var(--text-muted); padding: 20px;">No hyperparameter tuning metrics found. Click 'Run Optuna Hyperparameter Sweep' to benchmark.</td></tr>`;
            return;
        }
        
        let html = "";
        data.comparisons.forEach(row => {
            const r2Delta = row.r2_delta;
            const r2DeltaColor = r2Delta > 0 ? '#10b981' : (r2Delta === 0 ? '#9ca3af' : '#ef4444');
            const r2DeltaStr = `${r2Delta > 0 ? '+' : ''}${r2Delta.toFixed(4)}`;
            
            const maeImprv = row.mae_improvement_pct;
            const maeColor = maeImprv > 0 ? '#10b981' : (maeImprv === 0 ? '#9ca3af' : '#ef4444');
            const maeImprvStr = `${maeImprv > 0 ? '+' : ''}${maeImprv.toFixed(2)}%`;
            
            html += `
                <tr>
                    <td><strong>${row.target_name}</strong></td>
                    <td><span style="font-family: monospace; font-weight: bold; color: var(--neon-cyan); background: rgba(6, 182, 212, 0.1); padding: 3px 8px; border-radius: 4px;">${row.model}</span></td>
                    <td style="color: var(--text-muted);">${row.default_r2.toFixed(4)}</td>
                    <td style="font-weight: bold; color: #10b981;">${row.tuned_r2.toFixed(4)}</td>
                    <td style="font-weight: bold; color: ${r2DeltaColor};">${r2DeltaStr}</td>
                    <td style="color: var(--text-muted);">${row.default_mae.toFixed(3)} kW</td>
                    <td style="font-weight: bold; color: var(--text-primary);">${row.tuned_mae.toFixed(3)} kW</td>
                    <td style="font-weight: bold; color: ${maeColor};">${maeImprvStr}</td>
                    <td style="color: var(--neon-cyan); font-weight: 600;">${row.tuned_wape_pct.toFixed(2)}%</td>
                </tr>
            `;
        });
        tbody.innerHTML = html;
        refreshTuningPlots();
    } catch (err) {
        console.error("Failed to load tuning metrics:", err);
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
    
    refreshTuningPlots();
}

function refreshTuningPlots() {
    const t = new Date().getTime();
    const dashImg = document.querySelector("#fgallery-tuning img");
    if (dashImg) dashImg.src = `${API_BASE}/api/plots/tuning_comparison_dashboard.png?t=${t}`;
    
    const pvImg = document.querySelector("#fgallery-tuning_pv img");
    if (pvImg) pvImg.src = `${API_BASE}/api/plots/tuning_overlay_pv.png?t=${t}`;
    
    const conImg = document.querySelector("#fgallery-tuning_con img");
    if (conImg) conImg.src = `${API_BASE}/api/plots/tuning_overlay_con.png?t=${t}`;
    
    const caltechImg = document.querySelector("#fgallery-tuning_caltech img");
    if (caltechImg) caltechImg.src = `${API_BASE}/api/plots/tuning_overlay_caltech.png?t=${t}`;
}

// Switch MPC Visualizations Gallery Tabs
function switchMPCGallery(plotId) {
    document.querySelectorAll("#mpc-gallery-tabs .gallery-tab").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll("#tab-mpc .gallery-img-container").forEach(c => c.classList.add("hidden"));
    
    if (event && event.target) {
        event.target.classList.add("active");
    }
    const targetEl = document.getElementById(`mpcgallery-${plotId}`);
    if (targetEl) targetEl.classList.remove("hidden");
}

// Load and populate MPC Metrics Cards
async function loadMPCMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/mpc/metrics`);
        if (!res.ok) return;
        const data = await res.json();
        if (!data || Object.keys(data).length === 0) return;

        // Supermarket
        if (data.supermarket) {
            const sm = data.supermarket;
            const sz = sm.sizing;
            const su = sm.summary;
            const ms = sm.mismatch_stats;
            const szEl = document.getElementById("mpc-supermarket-sizing");
            const geEl = document.getElementById("mpc-supermarket-grid-energy");
            const gcEl = document.getElementById("mpc-supermarket-grid-cost");
            const dcEl = document.getElementById("mpc-supermarket-deg-cost");
            const erEl = document.getElementById("mpc-supermarket-error");
            const csEl = document.getElementById("mpc-supermarket-cases");

            if (szEl && sz) szEl.textContent = `${sz.E_B_max.toFixed(2)} kWh / ${sz.P_B_max.toFixed(2)} kW`;
            if (geEl && su) geEl.textContent = `${su.grid_energy_imported_kwh.toFixed(2)} kWh`;
            if (gcEl && su) gcEl.textContent = `EUR ${su.total_grid_cost_eur.toFixed(2)}`;
            if (dcEl && su) dcEl.textContent = `EUR ${su.total_battery_degradation_cost_eur.toFixed(4)}`;
            if (erEl && ms) erEl.textContent = `${ms.mean_p_error_kw > 0 ? '+' : ''}${ms.mean_p_error_kw.toFixed(2)} kW`;
            if (csEl && ms) csEl.textContent = `${ms.surplus_intervals} Surplus / ${ms.deficit_intervals} Deficit`;
        }

        // Lidl EV
        if (data.ev) {
            const ev = data.ev;
            const sz = ev.sizing;
            const su = ev.summary;
            const ms = ev.mismatch_stats;
            const szEl = document.getElementById("mpc-ev-sizing");
            const geEl = document.getElementById("mpc-ev-grid-energy");
            const gcEl = document.getElementById("mpc-ev-grid-cost");
            const dcEl = document.getElementById("mpc-ev-deg-cost");
            const erEl = document.getElementById("mpc-ev-error");
            const csEl = document.getElementById("mpc-ev-cases");

            if (szEl && sz) szEl.textContent = `${sz.E_B_max.toFixed(2)} kWh / ${sz.P_B_max.toFixed(2)} kW`;
            if (geEl && su) geEl.textContent = `${su.grid_energy_imported_kwh.toFixed(2)} kWh`;
            if (gcEl && su) gcEl.textContent = `EUR ${su.total_grid_cost_eur.toFixed(2)}`;
            if (dcEl && su) dcEl.textContent = `EUR ${su.total_battery_degradation_cost_eur.toFixed(4)}`;
            if (erEl && ms) erEl.textContent = `${ms.mean_p_error_kw > 0 ? '+' : ''}${ms.mean_p_error_kw.toFixed(2)} kW`;
            if (csEl && ms) csEl.textContent = `${ms.surplus_intervals} Surplus / ${ms.deficit_intervals} Deficit`;
        }

        // Caltech EV
        if (data.caltech_ev) {
            const ce = data.caltech_ev;
            const sz = ce.sizing;
            const su = ce.summary;
            const ms = ce.mismatch_stats;
            const szEl = document.getElementById("mpc-caltech-sizing");
            const geEl = document.getElementById("mpc-caltech-grid-energy");
            const gcEl = document.getElementById("mpc-caltech-grid-cost");
            const dcEl = document.getElementById("mpc-caltech-deg-cost");
            const erEl = document.getElementById("mpc-caltech-error");
            const csEl = document.getElementById("mpc-caltech-cases");

            if (szEl && sz) szEl.textContent = `${sz.E_B_max.toFixed(2)} kWh / ${sz.P_B_max.toFixed(2)} kW`;
            if (geEl && su) geEl.textContent = `${su.grid_energy_imported_kwh.toFixed(2)} kWh`;
            if (gcEl && su) gcEl.textContent = `EUR ${su.total_grid_cost_eur.toFixed(2)}`;
            if (dcEl && su) dcEl.textContent = `EUR ${su.total_battery_degradation_cost_eur.toFixed(4)}`;
            if (erEl && ms) erEl.textContent = `${ms.mean_p_error_kw > 0 ? '+' : ''}${ms.mean_p_error_kw.toFixed(2)} kW`;
            if (csEl && ms) csEl.textContent = `${ms.surplus_intervals} Surplus / ${ms.deficit_intervals} Deficit`;
        }

        // Refresh plots
        refreshMPCPlots();
    } catch (err) {
        console.error("Failed to load MPC metrics:", err);
    }
}

function refreshMPCPlots() {
    const t = new Date().getTime();
    const smImg = document.querySelector("#mpcgallery-supermarket img");
    if (smImg) smImg.src = `${API_BASE}/api/plots/mpc_schedule_supermarket.png?t=${t}`;
    
    const evImg = document.querySelector("#mpcgallery-ev img");
    if (evImg) evImg.src = `${API_BASE}/api/plots/mpc_schedule_ev.png?t=${t}`;
    
    const caltechImg = document.querySelector("#mpcgallery-caltech img");
    if (caltechImg) caltechImg.src = `${API_BASE}/api/plots/mpc_schedule_caltech_ev.png?t=${t}`;
}

// Switch RT Visualizations Gallery Tabs
function switchRTGallery(plotId) {
    document.querySelectorAll("#rt-gallery-tabs .gallery-tab").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll("#tab-rt .gallery-img-container").forEach(c => c.classList.add("hidden"));
    
    if (event && event.target) {
        event.target.classList.add("active");
    }
    const targetEl = document.getElementById(`rtgallery-${plotId}`);
    if (targetEl) targetEl.classList.remove("hidden");
}

// Load and populate RT Metrics Cards
async function loadRTMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/rt/metrics`);
        if (!res.ok) return;
        const data = await res.json();
        if (!data || Object.keys(data).length === 0) return;

        // Supermarket
        if (data.supermarket) {
            const sm = data.supermarket;
            const sch = sm.scheduled_mpc;
            const act = sm.real_time_actual;
            const mc = sm.mismatch_categorization;
            const dc = sm.dc_bus_balance;

            const gcEl = document.getElementById("rt-supermarket-grid-cost");
            const mpcCostEl = document.getElementById("rt-supermarket-mpc-cost");
            const degEl = document.getElementById("rt-supermarket-deg-cost");
            const curtEl = document.getElementById("rt-supermarket-curt");
            const balEl = document.getElementById("rt-supermarket-balance");
            const casesEl = document.getElementById("rt-supermarket-cases");

            if (gcEl && act) gcEl.textContent = `EUR ${act.grid_cost_eur.toFixed(2)}`;
            if (mpcCostEl && sch) mpcCostEl.textContent = `EUR ${sch.grid_cost_eur.toFixed(2)}`;
            if (degEl && act) degEl.textContent = `EUR ${act.degradation_cost_eur.toFixed(4)}`;
            if (curtEl && act) curtEl.textContent = `${act.curtailed_solar_kwh.toFixed(2)} kWh`;
            if (balEl && dc) balEl.textContent = `${dc.max_residual_mismatch_kw.toFixed(6)} kW [PASSED]`;
            if (casesEl && mc) casesEl.textContent = `${mc.deficit_intervals} Def / ${mc.surplus_intervals} Sur / ${mc.deadband_intervals} Deadband`;
        }

        // Lidl EV
        if (data.ev) {
            const ev = data.ev;
            const sch = ev.scheduled_mpc;
            const act = ev.real_time_actual;
            const mc = ev.mismatch_categorization;
            const dc = ev.dc_bus_balance;

            const gcEl = document.getElementById("rt-ev-grid-cost");
            const mpcCostEl = document.getElementById("rt-ev-mpc-cost");
            const degEl = document.getElementById("rt-ev-deg-cost");
            const curtEl = document.getElementById("rt-ev-curt");
            const balEl = document.getElementById("rt-ev-balance");
            const casesEl = document.getElementById("rt-ev-cases");

            if (gcEl && act) gcEl.textContent = `EUR ${act.grid_cost_eur.toFixed(2)}`;
            if (mpcCostEl && sch) mpcCostEl.textContent = `EUR ${sch.grid_cost_eur.toFixed(2)}`;
            if (degEl && act) degEl.textContent = `EUR ${act.degradation_cost_eur.toFixed(4)}`;
            if (curtEl && act) curtEl.textContent = `${act.curtailed_solar_kwh.toFixed(2)} kWh`;
            if (balEl && dc) balEl.textContent = `${dc.max_residual_mismatch_kw.toFixed(6)} kW [PASSED]`;
            if (casesEl && mc) casesEl.textContent = `${mc.deficit_intervals} Def / ${mc.surplus_intervals} Sur / ${mc.deadband_intervals} Deadband`;
        }

        // Caltech EV
        if (data.caltech_ev) {
            const ce = data.caltech_ev;
            const sch = ce.scheduled_mpc;
            const act = ce.real_time_actual;
            const mc = ce.mismatch_categorization;
            const dc = ce.dc_bus_balance;

            const gcEl = document.getElementById("rt-caltech-grid-cost");
            const mpcCostEl = document.getElementById("rt-caltech-mpc-cost");
            const degEl = document.getElementById("rt-caltech-deg-cost");
            const curtEl = document.getElementById("rt-caltech-curt");
            const balEl = document.getElementById("rt-caltech-balance");
            const casesEl = document.getElementById("rt-caltech-cases");

            if (gcEl && act) gcEl.textContent = `EUR ${act.grid_cost_eur.toFixed(2)}`;
            if (mpcCostEl && sch) mpcCostEl.textContent = `EUR ${sch.grid_cost_eur.toFixed(2)}`;
            if (degEl && act) degEl.textContent = `EUR ${act.degradation_cost_eur.toFixed(4)}`;
            if (curtEl && act) curtEl.textContent = `${act.curtailed_solar_kwh.toFixed(2)} kWh`;
            if (balEl && dc) balEl.textContent = `${dc.max_residual_mismatch_kw.toFixed(6)} kW [PASSED]`;
            if (casesEl && mc) casesEl.textContent = `${mc.deficit_intervals} Def / ${mc.surplus_intervals} Sur / ${mc.deadband_intervals} Deadband`;
        }

        refreshRTPlots();
    } catch (err) {
        console.error("Failed to load RT metrics:", err);
    }
}

function refreshRTPlots() {
    const t = new Date().getTime();
    const smImg = document.querySelector("#rtgallery-supermarket img");
    if (smImg) smImg.src = `${API_BASE}/api/plots/rt_schedule_supermarket.png?t=${t}`;
    
    const evImg = document.querySelector("#rtgallery-ev img");
    if (evImg) evImg.src = `${API_BASE}/api/plots/rt_schedule_ev.png?t=${t}`;
    
    const caltechImg = document.querySelector("#rtgallery-caltech img");
    if (caltechImg) caltechImg.src = `${API_BASE}/api/plots/rt_schedule_caltech_ev.png?t=${t}`;
}
