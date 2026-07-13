// Configuration
const API_BASE = ""; // Relative to server
let activePolls = {};
let systemState = {};

// On Load
document.addEventListener("DOMContentLoaded", () => {
    // Initial fetch of system status
    updateSystemStatus();
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
        updateCards(data);
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
function updateCards(state) {
    const scripts = ["ghi_getter", "pv_cleaner", "con_cleaner", "curtailment", "battery_optimizer"];
    
    scripts.forEach(key => {
        const card = document.getElementById(`card-${key}`);
        if (!card) return;
        
        const statusBadge = card.querySelector(".status-badge");
        const runBtn = card.querySelector(".btn-primary");
        const currentStatus = state[key].status;
        
        // Update Status Badge Class & Text
        statusBadge.className = `status-badge ${currentStatus}`;
        statusBadge.textContent = currentStatus;
        
        // Handle Active Run States
        if (currentStatus === "running") {
            card.classList.add("running");
            card.classList.remove("success", "error");
            runBtn.disabled = true;
            runBtn.textContent = "Executing...";
            
            // Start polling logs if not already doing so
            startLogPolling(key);
        } else {
            card.classList.remove("running");
            runBtn.disabled = false;
            runBtn.textContent = `Run ${getNiceName(key)}`;
            
            if (currentStatus === "success") {
                card.classList.add("success");
                card.classList.remove("error");
                // Stop polling and display output plots
                stopLogPolling(key);
                displayPlot(key);
            } else if (currentStatus === "error") {
                card.classList.add("error");
                card.classList.remove("success");
                stopLogPolling(key);
            }
        }
    });

    // Pipeline Specific Updates
    const pState = state["pipeline"];
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
        const stepsOrder = ["ghi_getter", "pv_cleaner", "con_cleaner", "curtailment", "battery_optimizer"];
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
            extractAndDisplaySizingMetrics(pState.logs);
            // Load all gallery images
            displayAllGalleryPlots();
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
    const nodes = ["ghi_getter", "pv_cleaner", "con_cleaner", "curtailment", "battery_optimizer"];
    
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
                const stepsOrder = ["ghi_getter", "pv_cleaner", "con_cleaner", "curtailment", "battery_optimizer"];
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
    
    const pollFunc = async () => {
        try {
            const res = await fetch(`${API_BASE}/api/logs?script=${scriptKey}`);
            if (!res.ok) throw new Error("Log fetch error");
            const data = await res.json();
            
            const logBox = document.getElementById(`log-${scriptKey}`);
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
        
        // Collapse other logs and expand the current one
        document.querySelectorAll(".console-log").forEach(box => {
            if (box.id !== `log-${scriptKey}`) {
                box.classList.add("collapsed");
                box.previousElementSibling.classList.add("collapsed");
            }
        });
        
        const currentBox = document.getElementById(`log-${scriptKey}`);
        currentBox.classList.remove("collapsed");
        currentBox.previousElementSibling.classList.remove("collapsed");
        
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
        "curtailment": "curtailment_results_2023.png",
        "battery_optimizer": "optimization_results.png"
    };
    return mapping[scriptKey];
}

// Load and display a plot in individual cards
function displayPlot(scriptKey) {
    const filename = getPlotFileName(scriptKey);
    if (!filename) return;
    
    const container = document.getElementById(`viz-${scriptKey}`);
    if (!container) return;
    
    // Add cache-busting timestamp
    const t = new Date().getTime();
    container.innerHTML = `<img src="${API_BASE}/api/plots/${filename}?t=${t}" alt="Plot Result" class="viz-img" onerror="imgError(this)">`;
}

// Load all plots into the gallery tabs
function displayAllGalleryPlots() {
    const t = new Date().getTime();
    
    document.querySelector("#gallery-pv img").src = `${API_BASE}/api/plots/data_cleaner_results.png?t=${t}`;
    document.querySelector("#gallery-con img").src = `${API_BASE}/api/plots/data_cleaner_con_results.png?t=${t}`;
    document.querySelector("#gallery-curtail img").src = `${API_BASE}/api/plots/curtailment_results_2023.png?t=${t}`;
    document.querySelector("#gallery-battery img").src = `${API_BASE}/api/plots/optimization_results.png?t=${t}`;
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
function extractAndDisplaySizingMetrics(logs) {
    const capacityMatch = logs.match(/Battery Capacity \(E_B_max\):\s*([\d\.]+)\s*kWh/i);
    const powerMatch = logs.match(/Battery Rated Power \(P_B_max\):\s*([\d\.]+)\s*kW/i);
    const costMatch = logs.match(/Total Annualized Cost \(CAPEX \+ OPEX\):\s*€\s*([\d\.,]+)/i);
    
    const capEl = document.getElementById("metric-capacity");
    const powEl = document.getElementById("metric-power");
    const costEl = document.getElementById("metric-cost");
    
    if (capacityMatch && capacityMatch[1]) {
        capEl.textContent = `${parseFloat(capacityMatch[1]).toFixed(2)} kWh`;
    } else {
        capEl.textContent = "-- kWh";
    }
    
    if (powerMatch && powerMatch[1]) {
        powEl.textContent = `${parseFloat(powerMatch[1]).toFixed(2)} kW`;
    } else {
        powEl.textContent = "-- kW";
    }
    
    if (costMatch && costMatch[1]) {
        costEl.textContent = `€${parseFloat(costMatch[1].replace(/,/g, '')).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
    } else {
        costEl.textContent = "-- €";
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
        "ghi_getter": "GHI Getter",
        "pv_cleaner": "PV Cleaner",
        "con_cleaner": "Consumption Cleaner",
        "curtailment": "Curtailment",
        "battery_optimizer": "Battery Optimizer"
    };
    return mapping[key] || key;
}
