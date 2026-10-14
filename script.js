async function predictRisk() {
const data = {
memory: document.getElementById("memory").value,
cpu: document.getElementById("cpu").value,
page_faults: document.getElementById("page_faults").value,
disk_io: document.getElementById("disk_io").value,
processes: document.getElementById("processes").value
};

```
for (const [key, value] of Object.entries(data)) {
    if (value === "" || !Number.isFinite(Number(value))) {
        alert("Please enter valid values in every field.");
        return;
    }
}

if (
    Number(data.memory) < 0 || Number(data.memory) > 100 ||
    Number(data.cpu) < 0 || Number(data.cpu) > 100 ||
    Number(data.disk_io) < 0 || Number(data.disk_io) > 100 ||
    Number(data.page_faults) < 0 ||
    Number(data.processes) < 1 ||
    !Number.isInteger(Number(data.processes))
) {
    alert("Please check the values and try again.");
    return;
}

document.getElementById("message").textContent =
    "Predicting risk...";

try {
    const response = await fetch("/predict", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify(data)
    });

    const result = await response.json();

    if (!response.ok) {
        throw new Error(result.error || "Prediction failed.");
    }

    document.getElementById("risk_score").textContent =
        result.risk_score + "%";

    document.getElementById("category").textContent =
        result.category;

    document.getElementById("message").textContent =
        result.category === "High"
            ? "High predicted risk. Prevention mechanisms will be developed next."
            : result.category === "Medium"
            ? "Moderate predicted risk. Continue monitoring."
            : "Low predicted risk. Continue normal monitoring.";

} catch (error) {
    document.getElementById("message").textContent =
        "Error: " + error.message;
}
```

}
