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
  const links = data.links || {};
  const finetune = data.finetune || {};
  const deployment = data.deployment || {};
  const qualitative = data.qualitative || {};

  const setText = (id, value) => {
    const element = document.getElementById(id);
    if (element) {
      element.textContent = value;
    }
  };

  const setImage = (id, src) => {
    const element = document.getElementById(id);
    if (element && src) {
      element.src = src;
    }
  };

  const setLink = (id, href) => {
    const element = document.getElementById(id);
    if (!element) {
      return;
    }
    if (href) {
      element.href = href;
      element.classList.remove("hidden");
    } else {
      element.classList.add("hidden");
    }
  };

  const formatSize = (value) => (value !== undefined && value !== null ? `${Number(value).toFixed(2)} MB` : null);
  const formatDiff = (label, value) => (value !== undefined && value !== null ? `${label} ${Number(value).toFixed(6)}` : null);

  setText("protocol-pill", meta.protocol_label || "Benchmark run");
  setText(
    "device-pill",
    meta.gpu_name ? `Recorded on ${meta.gpu_name}` : (meta.device ? `Recorded on ${String(meta.device).toUpperCase()}` : "Device unavailable")
  );
  setText("model-count", summary.model_count || "-");
  setText("best-accuracy", summary.best_accuracy_model ? summary.best_accuracy_model.name : "-");
  setText(
    "best-accuracy-note",
    summary.best_accuracy_model ? `${summary.best_accuracy_model.value} ${summary.best_accuracy_model.unit} on the held-out test split` : "-"
  );
  setText("fastest-model", summary.fastest_cpu_model ? summary.fastest_cpu_model.name : "-");
  setText(
    "fastest-model-note",
    summary.fastest_cpu_model ? `${summary.fastest_cpu_model.value} ${summary.fastest_cpu_model.unit} median CPU latency` : "-"
  );

  setText(
    "best-finetuned-model",
    summary.best_finetuned_model ? summary.best_finetuned_model.name : "Pending"
  );
  setText(
    "best-finetuned-note",
    summary.best_finetuned_model
      ? `${summary.best_finetuned_model.top1_accuracy}% top-1 after supervised adaptation`
      : "Fine-tune results pending"
  );

  setText(
    "protocol-description",
    meta.protocol_description ||
      "The current benchmark is a head-reset transfer baseline. Fine-tuning is implemented separately."
  );

  setImage("figure-accuracy", figures.accuracy_vs_latency);
  setImage("figure-footprint", figures.parameter_footprint);
  setImage("figure-latency", figures.latency_breakdown);

  const modelTable = document.getElementById("model-table");
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
    modelTable.appendChild(row);
  });

  const recommendedModels = document.getElementById("recommended-models");
  (summary.recommended_finetune_targets || []).forEach((modelKey) => {
    const match = models.find((model) => model.model_key === modelKey);
    const item = document.createElement("li");
    item.textContent = match ? `${match.name} - selected as a supervised adaptation candidate.` : modelKey;
    recommendedModels.appendChild(item);
  });
  if (!recommendedModels.children.length) {
    const item = document.createElement("li");
    item.textContent = "Fine-tune targets have not been selected yet.";
    recommendedModels.appendChild(item);
  }

  const finetuneTable = document.getElementById("finetune-table");
  if (finetune.available && Array.isArray(finetune.leaderboard) && finetune.leaderboard.length > 0) {
    finetune.leaderboard.forEach((entry) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${entry.name}</td>
        <td>${entry.top1_accuracy.toFixed(2)}%</td>
        <td>${entry.gain !== null && entry.gain !== undefined ? `+${entry.gain.toFixed(2)} pts` : "n/a"}</td>
      `;
      finetuneTable.appendChild(row);
    });
    setText(
      "finetune-summary-copy",
      `${finetune.leaderboard.length} fine-tuned model${finetune.leaderboard.length > 1 ? "s" : ""} completed. Supervised adaptation closes the transfer gap and produces a practical export candidate.`
    );
    setText(
      "export-target",
      finetune.deployment_candidate
        ? `Deployment target: ${finetune.deployment_candidate.name} — ${finetune.deployment_candidate.reason}`
        : (finetune.recommended_export_model ? `Deployment target: ${finetune.recommended_export_model}` : "Deployment target pending")
    );
  } else {
    const row = document.createElement("tr");
    row.innerHTML = "<td colspan='3'>Fine-tune artifacts not generated yet.</td>";
    finetuneTable.appendChild(row);
  }

  if (figures.finetune_accuracy_delta) {
    setImage("figure-finetune", figures.finetune_accuracy_delta);
    document.getElementById("finetune-figure-card")?.classList.remove("hidden");
  }

  if (qualitative.available && (figures.dataset_mosaic || figures.qualitative_before_after)) {
    setText(
      "qualitative-summary-copy",
      qualitative.summary || "Real EuroSAT test imagery before and after supervised adaptation."
    );
    setImage("figure-dataset-mosaic", figures.dataset_mosaic);
    setImage("figure-qualitative", figures.qualitative_before_after);
    document.getElementById("qualitative-section")?.classList.remove("hidden");
  }

  const artifactGrid = document.getElementById("artifact-grid");
  const renderArtifactCard = (title, payload) => {
    const card = document.createElement("article");
    card.className = "artifact-card";
    const normalizedStatus = String(payload?.status || "available").toLowerCase();
    const status = normalizedStatus === "ok" ? "available" : normalizedStatus;
    const badgeClass = status === "available" || status === "validated" ? "artifact-badge" : "artifact-badge warn";
    const descriptionBits = [];
    if (payload?.path) {
      descriptionBits.push(payload.path);
    }
    if (payload?.num_samples !== undefined) {
      descriptionBits.push(`${payload.num_samples} samples`);
    }
    if (payload?.size_mb !== undefined) {
      descriptionBits.push(formatSize(payload.size_mb));
    }
    if (payload?.reason) {
      descriptionBits.push(payload.reason);
    }
    if (payload?.backend) {
      descriptionBits.push(payload.backend);
    }
    const maxAbsDiff = formatDiff("max abs diff", payload?.max_abs_diff);
    const meanAbsDiff = formatDiff("mean abs diff", payload?.mean_abs_diff);
    if (maxAbsDiff) {
      descriptionBits.push(maxAbsDiff);
    }
    if (meanAbsDiff) {
      descriptionBits.push(meanAbsDiff);
    }
    card.innerHTML = `
      <span class="${badgeClass}">${status.toUpperCase()}</span>
      <strong>${title}</strong>
      <span class="artifact-meta">${descriptionBits.join(" • ") || "No artifact metadata available."}</span>
    `;
    if (payload?.path && (payload.path.startsWith("http") || payload.path.startsWith("assets/"))) {
      const link = document.createElement("a");
      link.href = payload.path;
      if (payload.path.startsWith("http")) {
        link.target = "_blank";
        link.rel = "noreferrer";
      }
      link.textContent = "Open artifact";
      card.appendChild(link);
    }
    return card;
  };

  if (deployment.available) {
    const selectedModel = deployment.selected_model || {};
    const visibleArtifacts = [
      ["FP32 ONNX", deployment.artifacts?.onnx_fp32],
      ["ONNX validation", deployment.artifacts?.onnx_validation ? { status: "validated", ...deployment.artifacts.onnx_validation } : null],
      ["Calibration data", deployment.artifacts?.calibration_data],
      ["INT8 ONNX", deployment.artifacts?.onnx_int8],
    ].filter(([, payload]) => payload && !["skipped", "pending", "missing"].includes(String(payload.status || "").toLowerCase()));

    setText(
      "deployment-copy",
      `The export pipeline packages ${selectedModel.name || "the selected model"} into validated FP32 and INT8 ONNX artifacts, together with reproducible calibration data for the quantization step.`
    );
    setText(
      "deployment-meta",
      `${selectedModel.name || "Selected model"} • ${visibleArtifacts.length} generated deployment artifacts`
    );
    visibleArtifacts.forEach(([title, payload]) => {
      artifactGrid.appendChild(renderArtifactCard(title, payload || {}));
    });
  } else {
    setText("deployment-meta", "Deployment export pending");
    artifactGrid.appendChild(
      renderArtifactCard("Deployment export", {
        status: "pending",
        reason: "Run 04_export_deployment_artifacts.py after fine-tuning finishes.",
      })
    );
  }

  setLink("repo-link", links.repo_url);
  setLink("pdf-link", links.pdf_path);
  setLink("readme-link", links.readme_url);
  setLink("deliverable-repo", links.repo_url);
  setLink("deliverable-pages", links.pages_url);
  setLink("deliverable-pdf", links.pdf_path);
  setLink("footer-repo", links.repo_url);
  setLink("footer-readme", links.readme_url);
  setLink("footer-pdf", links.pdf_path);

  const generatedAt = data.generated_at ? new Date(data.generated_at) : null;
  const footerBits = [
    meta.dataset || "EuroSAT",
    meta.split_counts ? `train ${meta.split_counts.train}, val ${meta.split_counts.val}, test ${meta.split_counts.test}` : null,
    generatedAt ? `generated ${generatedAt.toLocaleString()}` : null,
  ].filter(Boolean);
  setText("footer-meta", footerBits.join(" • "));
})();
