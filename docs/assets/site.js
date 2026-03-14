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

  setText("protocol-pill", meta.protocol_label || "Benchmark run");
  setText("device-pill", meta.device ? `Recorded on ${String(meta.device).toUpperCase()}` : "Device unavailable");
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

  const jobHighlights = document.getElementById("job-highlights");
  (data.job_alignment?.highlights || []).forEach((highlight) => {
    const item = document.createElement("li");
    item.textContent = highlight;
    jobHighlights.appendChild(item);
  });

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
    item.textContent = match ? `${match.name} — strong candidate for supervised adaptation.` : modelKey;
    recommendedModels.appendChild(item);
  });

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
      `${finetune.leaderboard.length} fine-tuned model${finetune.leaderboard.length > 1 ? "s" : ""} completed. The strongest checkpoint is ready for deployment export.`
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

  const artifactGrid = document.getElementById("artifact-grid");
  const renderArtifactCard = (title, payload) => {
    const card = document.createElement("article");
    card.className = "artifact-card";
    const status = payload?.status || "available";
    const badgeClass = status === "ok" || status === "available" ? "artifact-badge" : "artifact-badge warn";
    const descriptionBits = [];
    if (payload?.path) {
      descriptionBits.push(payload.path);
    }
    if (payload?.size_mb !== undefined) {
      descriptionBits.push(`${payload.size_mb} MB`);
    }
    if (payload?.reason) {
      descriptionBits.push(payload.reason);
    }
    if (payload?.backend) {
      descriptionBits.push(payload.backend);
    }
    if (payload?.max_abs_diff !== undefined) {
      descriptionBits.push(`max abs diff ${payload.max_abs_diff}`);
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
    const deploymentMeta = deployment.meta || {};
    setText(
      "deployment-copy",
      `The export pipeline packages ${selectedModel.name || "the selected model"} into ONNX, prepares calibration data, and attempts TensorRT engine builds when local tooling is available.`
    );
    setText(
      "deployment-meta",
      `${selectedModel.name || "Selected model"} • ONNX runtime ${deploymentMeta.onnxruntime_available ? "available" : "missing"} • TensorRT tooling ${deploymentMeta.trtexec_available ? "available" : "missing"}`
    );

    const artifactOrder = [
      ["FP32 ONNX", deployment.artifacts?.onnx_fp32],
      ["ONNX validation", deployment.artifacts?.onnx_validation],
      ["Calibration data", deployment.artifacts?.calibration_data],
      ["INT8 ONNX", deployment.artifacts?.onnx_int8],
      ["TensorRT FP16 engine", deployment.artifacts?.tensorrt_fp16],
      ["TensorRT INT8 engine", deployment.artifacts?.tensorrt_int8],
    ];
    artifactOrder.forEach(([title, payload]) => {
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
  setLink("report-link", links.report_url);
  setLink("pdf-link", links.pdf_path);
  setLink("pages-link", links.pages_url);
  setLink("readme-link", links.readme_url);
  setLink("deliverable-repo", links.repo_url);
  setLink("deliverable-report", links.report_url);
  setLink("deliverable-pages", links.pages_url);
  setLink("deliverable-pdf", links.pdf_path);
  setLink("footer-repo", links.repo_url);
  setLink("footer-report", links.report_url);
  setLink("job-link", data.job_alignment?.role_url);

  const generatedAt = data.generated_at ? new Date(data.generated_at) : null;
  const footerBits = [
    meta.dataset || "EuroSAT",
    meta.split_counts ? `train ${meta.split_counts.train}, val ${meta.split_counts.val}, test ${meta.split_counts.test}` : null,
    generatedAt ? `generated ${generatedAt.toLocaleString()}` : null,
  ].filter(Boolean);
  setText("footer-meta", footerBits.join(" • "));
})();
