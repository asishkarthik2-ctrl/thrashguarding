async function refreshMonitor() {
    try {
        const response = await fetch("/predict", { cache: "no-store" });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Prediction failed.");
        }

        const snapshot = result.snapshot || {};
        document.getElementById("cpu_percent").textContent = `${snapshot.cpu_percent ?? "--"}%`;
        document.getElementById("memory_percent").textContent = `${snapshot.memory_percent ?? "--"}%`;
        document.getElementById("free_memory_percent").textContent = `${snapshot.free_memory_percent ?? "--"}%`;
        document.getElementById("swap_percent").textContent = `${snapshot.swap_percent ?? "--"}%`;
        document.getElementById("page_faults_per_sec").textContent = `${snapshot.page_faults_per_sec ?? "--"}/s`;
        document.getElementById("hard_faults_per_sec").textContent = `${snapshot.hard_faults_per_sec ?? "--"}/s`;
        document.getElementById("process_count").textContent = snapshot.process_count ?? "--";
        document.getElementById("timestamp").textContent = snapshot.timestamp ?? "--";
        document.getElementById("telemetry_status").textContent = result.telemetry_status ?? "unavailable";

        document.getElementById("risk_score").textContent = result.risk_score === null ? "--" : `${result.risk_score}%`;
        document.getElementById("category").textContent = result.category;
        document.getElementById("message").textContent = result.risk_score === null
            ? "No readable process samples are available for the model."
            : "Live process estimates refreshed. The token booth is advisory only and does not change OS scheduling.";

        const processRiskList = document.getElementById("process_risk_list");
        processRiskList.replaceChildren();
        for (const process of result.process_predictions || []) {
            const item = document.createElement("li");
            item.textContent = `${process.process_name} (PID ${process.pid}): ${process.risk_score}% ${process.risk_level} risk, ${process.rss_mb} MB resident`;
            processRiskList.appendChild(item);
        }
        if (!processRiskList.childElementCount) {
            processRiskList.innerHTML = "<li>No process predictions available.</li>";
        }

        const booth = result.token_booth || {};
        const tokenList = document.getElementById("token_booth_list");
        tokenList.replaceChildren();
        document.getElementById("token_summary").textContent = `Execution order: lowest risk first | 10 tokens per process | ${booth.risk_classification || "unavailable"} overall | ${booth.refill_mode || "no refill"} refill | simulation only`;
        for (const process of booth.processes || []) {
            const item = document.createElement("li");
            item.textContent = `Priority ${process.priority_rank}: ${process.process_name} (PID ${process.pid}), ${process.risk_score}% ${process.risk_level} risk, ${process.tokens_remaining}/${process.tokens_total} tokens, refill ${process.refill_mode}`;
            tokenList.appendChild(item);
        }
        if (!tokenList.childElementCount) {
            tokenList.innerHTML = "<li>No token rows available.</li>";
        }
    } catch (error) {
        document.getElementById("message").textContent = `Monitor error: ${error.message}`;
    }
}

document.addEventListener("DOMContentLoaded", () => {
    document.getElementById("refreshButton").addEventListener("click", refreshMonitor);
    refreshMonitor();
    setInterval(refreshMonitor, 3000);
});
