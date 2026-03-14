(function () {
  const data = window.BENCHMARK_SITE_DATA;

  if (!data) {
    document.body.innerHTML = "<main style='padding:24px;font-family:sans-serif'>Benchmark site data was not found. Run python 02_generate_visualizations.py first.</main>";
    return;
  }

  const summary = data.summary || {};
  const meta = data.meta || {};
  const models = data.models || [];
  const figures = data.figures || {};

  const setText = (id, value) => {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = value;
    }
  };

  const setImage = (id, src) => {
    const element = document.getElementById(id);
    if (element) {
      element.src = src;
    }
  };

  setText("protocol-pill", meta.protocol_label || "Benchmark run");
  setText("device-pill", meta.device ? `Recorded on ${meta.device.toUpperCase()}` : "Device unavailable");
  setText("model-count", summary.model_count || "-");
  setText(
    "best-accuracy",
    summary.best_accuracy_model ? `${summary.best_accuracy_model.name}` : "-"
  );
  setText(
    "best-accuracy-note",
    summary.best_accuracy_model ? `${summary.best_accuracy_model.value} ${summary.best_accuracy_model.unit} on the held-out test split` : "-"
  );
  setText(
    "fastest-model",
    summary.fastest_cpu_model ? `${summary.fastest_cpu_model.name}` : "-"
  );
  setText(
    "fastest-model-note",
    summary.fastest_cpu_model ? `${summary.fastest_cpu_model.value} ${summary.fastest_cpu_model.unit} median CPU latency` : "-"
  );
  setText(
    "efficiency-model",
    summary.best_size_efficiency_model ? `${summary.best_size_efficiency_model.name}` : "-"
  );
  setText(
    "efficiency-model-note",
    summary.best_size_efficiency_model ? `${summary.best_size_efficiency_model.value} ${summary.best_size_efficiency_model.unit}` : "-"
  );

  setText(
    "protocol-description",
    meta.protocol_description ||
      "The current benchmark is a head-reset transfer baseline. Fine-tuning is implemented separately."
  );

  setImage("figure-accuracy", figures.accuracy_vs_latency);
  setImage("figure-footprint", figures.parameter_footprint);
  setImage("figure-latency", figures.latency_breakdown);

  const jobHighlights = document.getElementById("job-highlights");
  (data.job_alignment?.highlights || []).forEach((highlight) => {
    const item = document.createElement("li");
    item.textContent = highlight;
    jobHighlights.appendChild(item);
  });

  const table = document.getElementById("model-table");
  models.forEach((model) => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${model.name}</td>
      <td>${model.top1_accuracy.toFixed(2)}%</td>
      <td>${model.cpu_latency_ms.toFixed(2)} ms</td>
      <td>${model.gpu_latency_ms !== null && model.gpu_latency_ms !== undefined ? `${model.gpu_latency_ms.toFixed(2)} ms` : "n/a"}</td>
      <td>${model.params_m.toFixed(1)}M</td>
      <td>${model.macs_m !== null && model.macs_m !== undefined ? `${model.macs_m.toFixed(0)}M` : "n/a"}</td>
      <td>${model.throughput_bs1 !== null && model.throughput_bs1 !== undefined ? `${model.throughput_bs1} img/s` : "n/a"}</td>
    `;
    table.appendChild(row);
  });

  const recommendedModels = document.getElementById("recommended-models");
  const recommended = summary.recommended_finetune_targets || [];
  recommended.forEach((modelKey) => {
    const match = models.find((model) => model.model_key === modelKey);
    const item = document.createElement("li");
    item.textContent = match ? `${match.name} — next candidate for supervised adaptation.` : modelKey;
    recommendedModels.appendChild(item);
  });

  const generatedAt = data.generated_at ? new Date(data.generated_at) : null;
  const footerBits = [
    meta.dataset || "EuroSAT",
    meta.split_counts ? `train ${meta.split_counts.train}, val ${meta.split_counts.val}, test ${meta.split_counts.test}` : null,
    generatedAt ? `generated ${generatedAt.toLocaleString()}` : null,
  ].filter(Boolean);
  setText("footer-meta", footerBits.join(" • "));

  const jobLink = document.getElementById("job-link");
  if (jobLink && data.job_alignment?.role_url) {
    jobLink.href = data.job_alignment.role_url;
  }
})();
