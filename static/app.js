const $ = (id) => document.getElementById(id);
const LABELS = ["negative", "neutral", "positive"];
const HEX = { negative: "#FF7A93", neutral: "#F2C14E", positive: "#4BD6A0" };
const MODEL_NAME = "TF-IDF (words + characters) with Logistic Regression";
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const pct = (x) => `${(x * 100).toFixed(1)}%`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));

let model = null;     // trained-model metrics, loaded once
let current = null;   // last analysis
let fileName = "";
let filter = "all";

function setStatus(msg, isError = false) {
  $("status").textContent = msg;
  $("status").classList.toggle("error", isError);
}

/* ---------- Model panel (static, from training) ---------- */

async function loadModel() {
  try {
    model = await (await fetch("/api/model")).json();
  } catch {
    $("modelChip").textContent = "Model unavailable";
    return;
  }
  $("modelChip").textContent = `Model ready · ${pct(model.accuracy)} held-out accuracy`;
  $("factTrain").textContent = model.total_reviews.toLocaleString();
  $("factAcc").textContent = pct(model.accuracy);

  $("kpis").innerHTML = [
    ["Accuracy", pct(model.accuracy)],
    ["Macro F1", model.macro_f1.toFixed(3)],
    ["Test reviews", model.test_size.toLocaleString()],
  ].map(([k, v]) => `<div class="kpi"><dt>${k}</dt><dd>${v}</dd></div>`).join("");

  $("perClass").innerHTML = LABELS.map((l) => {
    const p = model.per_class[l];
    return `<tr>
      <td><span class="pill ${l}">${l}</span></td>
      <td>${p.precision.toFixed(3)}</td><td>${p.recall.toFixed(3)}</td><td>${p.f1.toFixed(3)}</td>
    </tr>`;
  }).join("");

  $("note").textContent = `Training data: ${model.dataset}, a generated corpus. Accuracy on real reviews will be lower. `
    + "Retrain on your own labelled CSV with python train.py --data your_file.csv.";

  const cm = model.confusion;
  const rowTotals = cm.map((r) => r.reduce((a, b) => a + b, 0));
  let html = `<div class="cm"><span></span>${LABELS.map((l) => `<span class="cm-h">${cap(l)}</span>`).join("")}`;
  cm.forEach((row, i) => {
    html += `<span class="cm-r">${cap(LABELS[i])}</span>`;
    row.forEach((v, j) => {
      const share = (v / rowTotals[i]).toFixed(3);
      html += `<span class="cm-c ${i === j ? "diag" : ""}" style="--a:${share}"
        title="True ${LABELS[i]}, predicted ${LABELS[j]}: ${v}">${v.toLocaleString()}</span>`;
    });
  });
  $("matrix").innerHTML = html + "</div>";
}

/* ---------- Analysis ---------- */

async function analyze(file) {
  if (!file.name.toLowerCase().endsWith(".csv")) {
    setStatus("Please choose a .csv file.", true);
    return;
  }
  setStatus("Scoring reviews…");
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/analyze", { method: "POST", body: form });
    const body = await res.json();
    if (!res.ok) {
      setStatus(body.detail || "Something went wrong.", true);
      return;
    }
    setStatus("");
    fileName = file.name;
    render(body);
  } catch {
    setStatus("Could not reach the server. Is it running?", true);
  }
}

function render(d) {
  current = d;
  filter = "all";
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("active", b.dataset.f === "all"));

  const extras = [`${fileName}`, `${d.total.toLocaleString()} reviews`];
  if (d.user_accuracy !== null) extras.push(`${d.user_accuracy}% match your labels`);
  if (d.star_agreement !== null) extras.push(`${d.star_agreement}% agree with stars`);
  $("subtitle").textContent = extras.join(" · ");

  drawSpectrum(d.reviews, d.mean_score);

  $("split").innerHTML = d.distribution.map((x) => `
    <article class="split-item ${x.label}">
      <p class="split-label">${cap(x.label)}</p>
      <p class="split-value">${x.count.toLocaleString()}</p>
      <p class="split-sub">${x.percent}% of reviews</p>
      <div class="split-bar"><i style="width:${x.percent}%"></i></div>
    </article>`).join("");

  $("words").innerHTML = LABELS.map((l) => `
    <div class="words-col ${l}">
      <h4>${cap(l)}</h4>
      <div class="chips">${d.top_words[l].map((w) => `<span class="chip-word">${esc(w)}</span>`).join("")}</div>
    </div>`).join("");

  renderTable();
  $("results").hidden = false;
  $("results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function drawSpectrum(reviews, mean) {
  const W = 960, H = 230, P = 28;
  const x = (s) => P + ((s + 1) / 2) * (W - 2 * P);
  const band = (W - 2 * P) / 3;
  let out = "";
  LABELS.forEach((lab, i) => {
    out += `<rect x="${P + i * band}" y="0" width="${band}" height="${H}" fill="${HEX[lab]}" opacity="0.07"/>`;
  });
  out += `<line x1="${P}" x2="${W - P}" y1="${H / 2}" y2="${H / 2}" stroke="#2A2F3A"/>`;

  // Draw at most ~1,600 dots so the page stays fast on big files.
  const step = Math.max(1, Math.ceil(reviews.length / 1600));
  reviews.forEach((r, i) => {
    if (i % step !== 0) return;
    const spread = ((i * 0.618034) % 1) - 0.5;   // stable vertical spread
    const cy = H / 2 + spread * (H - 50);
    out += `<circle cx="${x(r.score).toFixed(1)}" cy="${cy.toFixed(1)}" r="3.4" fill="${HEX[r.sentiment]}" fill-opacity="0.85">
      <title>${esc(r.sentiment)} (${Math.round(r.confidence * 100)}%): ${esc(r.text.slice(0, 140))}</title></circle>`;
  });

  out += `<line x1="${x(mean)}" x2="${x(mean)}" y1="10" y2="${H - 10}" stroke="#F4F5F7" stroke-width="1.5" stroke-dasharray="4 5"/>`;
  $("spectrum").innerHTML = out;
  $("meanLabel").textContent = `Average ${mean > 0 ? "+" : ""}${mean.toFixed(2)}`;
}

function renderTable() {
  const rows = current.reviews.filter((r) => filter === "all" || r.sentiment === filter);
  const shown = rows.slice(0, 300);
  $("rows").innerHTML = shown.length
    ? shown.map((r) => `
        <tr>
          <td class="text-cell">${esc(r.text)}</td>
          <td><span class="pill ${r.sentiment}">${r.sentiment}</span></td>
          <td><span class="conf"><i style="width:${Math.round(r.confidence * 100)}%"></i></span><span class="conf-num">${Math.round(r.confidence * 100)}%</span></td>
          <td>${r.stars ?? "—"}</td>
        </tr>`).join("")
    : `<tr><td colspan="4" class="muted">No reviews in this group.</td></tr>`;
  $("rowcount").textContent = `Showing ${shown.length} of ${rows.length.toLocaleString()} reviews`;
}

/* ---------- Downloads ---------- */

function download(filename, content, type) {
  const url = URL.createObjectURL(new Blob([content], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

const stamp = () => new Date().toISOString().slice(0, 10);
const csvCell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;

function downloadCSV() {
  if (!current) return;
  const lines = [["review", "sentiment", "confidence", "score", "stars", "matches_stars"].join(",")];
  for (const r of current.reviews) {
    const match = r.expected == null ? "" : (r.expected === r.sentiment ? "yes" : "no");
    lines.push([r.text, r.sentiment, r.confidence, r.score, r.stars ?? "", match].map(csvCell).join(","));
  }
  download(`sentimentx-reviews-${stamp()}.csv`, "\uFEFF" + lines.join("\n"), "text/csv;charset=utf-8");
}

function downloadJSON() {
  if (!current) return;
  const report = {
    product: "SentimentX - The Sentiment Analysis using Machine Learning",
    generated_at: new Date().toISOString(),
    source_file: fileName,
    model: MODEL_NAME,
    training_data: model ? model.dataset : null,
    model_metrics: model,
    total_reviews: current.total,
    mean_score: current.mean_score,
    user_label_accuracy_percent: current.user_accuracy,
    star_agreement_percent: current.star_agreement,
    distribution: current.distribution,
    top_words: current.top_words,
    reviews: current.reviews,
  };
  download(`sentimentx-report-${stamp()}.json`, JSON.stringify(report, null, 2), "application/json");
}

/* ---------- Wiring ---------- */

$("pick").onclick = () => $("file").click();
$("file").onchange = (e) => e.target.files[0] && analyze(e.target.files[0]);

$("sample").onclick = async () => {
  const blob = await (await fetch("sample_reviews.csv")).blob();
  analyze(new File([blob], "sample_reviews.csv", { type: "text/csv" }));
};

const drop = $("drop");
["dragenter", "dragover"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) =>
  drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file) analyze(file);
});

document.querySelectorAll("#tabs button").forEach((b) => {
  b.onclick = () => {
    filter = b.dataset.f;
    document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("active", x === b));
    renderTable();
  };
});

$("dl-csv").onclick = downloadCSV;
$("dl-json").onclick = downloadJSON;
$("reset").onclick = () => {
  $("results").hidden = true;
  $("file").value = "";
  current = null;
  setStatus("");
  window.scrollTo({ top: 0, behavior: "smooth" });
};

loadModel();
