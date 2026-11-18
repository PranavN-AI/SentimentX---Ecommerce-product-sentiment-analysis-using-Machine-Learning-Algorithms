const $ = (id) => document.getElementById(id);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const LABELS = ["positive", "neutral", "negative"];
const HEX = { positive: "#4BD6A0", neutral: "#F2C14E", negative: "#FF7A93" };
const MODEL_NAME = "TF-IDF (words + characters) with calibrated Linear SVM";
const cap = (s) => s[0].toUpperCase() + s.slice(1);
const pct = (x, d = 1) => `${Number(x).toFixed(d)}%`;
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[c]));
const stamp = () => new Date().toISOString().slice(0, 10);

const VIEW_TITLES = {
  upload: ["Upload reviews", "Analyze a CSV of product reviews"],
  overview: ["Overview", ""],
  issues: ["Customer issues", "Why customers are not satisfied"],
  reviews: ["Reviews", "Read and filter every review"],
  model: ["Model quality", "How accurate the model is on unseen reviews"],
  export: ["Export", "Download the analysis"],
};

let analysis = null;   // last analysis from the server
let fileName = "";
let modelInfo = null;  // model metrics, loaded once
let filter = "all";
let view = "upload";

/* ---------- Views and navigation ---------- */

function showView(name) {
  if (!VIEW_TITLES[name]) return;
  view = name;
  $$(".view").forEach((v) => { v.hidden = v.dataset.view !== name; });
  $$("[data-nav] [data-view]").forEach((b) => {
    if (b.dataset.view === name) b.setAttribute("aria-current", "page");
    else b.removeAttribute("aria-current");
  });
  const [title, sub] = VIEW_TITLES[name];
  $("pageTitle").textContent = title;
  $("pageSub").textContent = name === "overview" && analysis
    ? `${fileName} · ${analysis.total.toLocaleString()} reviews`
    : sub;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setNavEnabled(enabled) {
  $$("[data-nav] [data-view]").forEach((b) => {
    if (b.dataset.view !== "model" && b.dataset.view !== "upload") b.disabled = !enabled;
  });
  $("newTop").hidden = !enabled;
  $("newSide").hidden = !enabled;
}

function toast(msg, isError = false) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.toggle("error", isError);
  t.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { t.hidden = true; }, 3600);
}

function setStatus(msg, isError = false) {
  $("status").textContent = msg;
  $("status").classList.toggle("error", isError);
}

/* ---------- Model ---------- */

async function loadModel() {
  try {
    modelInfo = await (await fetch("/api/model")).json();
  } catch {
    $("modelChip").textContent = "Model unavailable";
    return;
  }
  $("modelChip").textContent = `Model ready · ${pct(modelInfo.accuracy * 100, 2)} accuracy`;
  renderModelView();
}

function renderModelView() {
  if (!modelInfo) return;
  const m = modelInfo;
  $("modelKpis").innerHTML = [
    ["Accuracy", pct(m.accuracy * 100, 2), `on ${m.test_size.toLocaleString()} unseen reviews`, "var(--accent)"],
    ["Macro F1", m.macro_f1.toFixed(3), "average across classes", "var(--accent)"],
    ["Training reviews", m.train_size.toLocaleString(), "used to fit the model", "var(--ink)"],
    ["Total reviews", m.total_reviews.toLocaleString(), "in the training dataset", "var(--ink)"],
  ].map(([k, v, s, c]) => `<div class="kpi" style="--c:${c}"><p class="kpi-label">${k}</p><p class="kpi-value">${v}</p><p class="kpi-sub">${s}</p></div>`).join("");

  $("perClass").innerHTML = `<thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Reviews</th></tr></thead><tbody>${
    LABELS.map((l) => {
      const p = m.per_class[l];
      return `<tr><td><span class="pill ${l}">${l}</span></td><td>${p.precision.toFixed(3)}</td><td>${p.recall.toFixed(3)}</td><td>${p.f1.toFixed(3)}</td><td>${p.support.toLocaleString()}</td></tr>`;
    }).join("")}</tbody>`;

  const cm = m.confusion;
  const rowTotals = cm.map((r) => r.reduce((a, b) => a + b, 0));
  let html = `<div class="cm"><span></span>${LABELS.map((l) => `<span class="cm-h">${cap(l)}</span>`).join("")}`;
  cm.forEach((row, i) => {
    html += `<span class="cm-r">${cap(LABELS[i])}</span>`;
    row.forEach((v, j) => {
      html += `<span class="cm-c ${i === j ? "diag" : ""}" style="--a:${(v / rowTotals[i]).toFixed(3)}"
        title="True ${LABELS[i]}, predicted ${LABELS[j]}: ${v}">${v.toLocaleString()}</span>`;
    });
  });
  $("matrix").innerHTML = html + "</div>";

  $("modelNote").textContent = `Model: ${m.model || MODEL_NAME}. Training data: ${m.dataset}. `
    + "Metrics come from a 15% split of the dataset that the model never trained on. "
    + "To use your own labelled data, run python train.py --data your_file.csv.";
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
    analysis = body;
    fileName = file.name;
    renderAll();
    setNavEnabled(true);
    showView("overview");
    toast(`Analyzed ${analysis.total.toLocaleString()} reviews`);
  } catch {
    setStatus("Could not reach the server. Is it running?", true);
  }
}

function renderAll() {
  renderOverview();
  renderIssues();
  renderReviews();
}

function renderOverview() {
  const d = analysis;
  const byLabel = Object.fromEntries(d.distribution.map((x) => [x.label, x]));
  const cards = [
    ["Reviews analysed", d.total.toLocaleString(), fileName, "var(--ink)"],
    ...LABELS.map((l) => [cap(l), pct(byLabel[l].percent), `${byLabel[l].count.toLocaleString()} reviews`, HEX[l]]),
  ];
  if (d.star_agreement !== null) cards.push(["Agrees with stars", pct(d.star_agreement), "of reviews with a rating", "var(--accent)"]);
  if (d.user_accuracy !== null) cards.push(["Matches your labels", pct(d.user_accuracy), "of labelled reviews", "var(--accent)"]);
  $("kpis").innerHTML = cards.map(([k, v, s, c]) =>
    `<div class="kpi" style="--c:${c}"><p class="kpi-label">${k}</p><p class="kpi-value">${v}</p><p class="kpi-sub">${esc(s)}</p></div>`).join("");

  $("avgLabel").textContent = `Average score ${d.mean_score > 0 ? "+" : ""}${d.mean_score.toFixed(2)}`;
  drawSpectrum(d.reviews, d.mean_score);

  $("stackbar").innerHTML = LABELS.map((l) =>
    `<i class="${l}" style="width:${byLabel[l].percent}%" title="${cap(l)}: ${pct(byLabel[l].percent)}"></i>`).join("");
  $("legend").innerHTML = LABELS.map((l) =>
    `<span class="${l}">${cap(l)} · ${pct(byLabel[l].percent)}</span>`).join("");

  const top = d.issues.issues[0];
  const existing = $("topIssueCard");
  if (existing) existing.remove();
  const card = document.createElement("div");
  card.className = "card";
  card.id = "topIssueCard";
  card.innerHTML = top
    ? `<div class="card-head"><div><h3>Top customer issue</h3><p class="muted small">Most common complaint in negative reviews</p></div>
         <button class="btn ghost" data-goto="issues">See all issues</button></div>
       <p class="issue-hero"><strong>${esc(top.issue)}</strong> is mentioned in ${pct(top.percent)} of negative reviews (${top.reviews.toLocaleString()} reviews).</p>`
    : `<div class="card-head"><div><h3>Top customer issue</h3><p class="muted small">No negative reviews were found in this file.</p></div></div>`;
  // Place the card in the overview section, after the split card.
  $("kpis").parentElement.appendChild(card);
  card.querySelector("[data-goto]")?.addEventListener("click", () => showView("issues"));
}

function renderIssues() {
  const d = analysis.issues;
  const share = analysis.total ? (d.base_reviews / analysis.total * 100) : 0;
  $("issuesSub").textContent = d.base_reviews
    ? `Based on ${d.base_reviews.toLocaleString()} negative reviews (${pct(share)} of all reviews).`
    : "No negative reviews were found in this file.";
  if (!d.issues.length) {
    $("issueList").innerHTML = `<li class="muted">No recurring complaint themes found.</li>`;
    return;
  }
  const maxPct = Math.max(...d.issues.map((i) => i.percent), 1);
  $("issueList").innerHTML = d.issues.map((i, idx) => `
    <li><details class="issue" ${idx === 0 ? "open" : ""}>
      <summary>
        <span class="rank">${idx + 1}</span>
        <div class="issue-main">
          <div class="issue-head-row">
            <span class="issue-title">${esc(i.issue)}</span>
            <span class="issue-meta">${i.reviews.toLocaleString()} · ${pct(i.percent)}</span>
          </div>
          <div class="bar"><i style="width:${(i.percent / maxPct * 100).toFixed(1)}%"></i></div>
        </div>
      </summary>
      <ul class="examples">${i.examples.map((e) => `<li>“${esc(e)}”</li>`).join("")}</ul>
    </details></li>`).join("");
}

function drawSpectrum(reviews, mean) {
  const W = 960, P = 26;
  const H = window.innerWidth < 640 ? 320 : 220;   // taller on phones so the dots are easier to see
  $("spectrum").setAttribute("viewBox", `0 0 ${W} ${H}`);
  const x = (s) => P + ((s + 1) / 2) * (W - 2 * P);
  const band = (W - 2 * P) / 3;
  let out = "";
  LABELS.forEach((lab, i) => {
    out += `<rect x="${P + i * band}" y="0" width="${band}" height="${H}" fill="${HEX[lab]}" opacity="0.07"/>`;
  });
  out += `<line x1="${P}" x2="${W - P}" y1="${H / 2}" y2="${H / 2}" stroke="#2A3142"/>`;
  const step = Math.max(1, Math.ceil(reviews.length / 1600));
  reviews.forEach((r, i) => {
    if (i % step !== 0) return;
    const spread = ((i * 0.618034) % 1) - 0.5;
    const cy = H / 2 + spread * (H - 46);
    out += `<circle cx="${x(r.score).toFixed(1)}" cy="${cy.toFixed(1)}" r="3.3" fill="${HEX[r.sentiment]}" fill-opacity="0.85">
      <title>${esc(cap(r.sentiment))} (${Math.round(r.confidence * 100)}%): ${esc(r.text.slice(0, 140))}</title></circle>`;
  });
  out += `<line x1="${x(mean)}" x2="${x(mean)}" y1="10" y2="${H - 10}" stroke="#F4F5F7" stroke-width="1.5" stroke-dasharray="4 5"/>`;
  $("spectrum").innerHTML = out;
}

function renderReviews() {
  const q = $("search").value.trim().toLowerCase();
  const rows = analysis.reviews.filter((r) =>
    (filter === "all" || r.sentiment === filter) && (!q || r.text.toLowerCase().includes(q)));
  const shown = rows.slice(0, 300);
  $("reviewList").innerHTML = shown.length
    ? shown.map((r) => `
      <article class="review">
        <p class="review-text">${esc(r.text)}</p>
        <div class="review-meta">
          <span class="pill ${r.sentiment}">${r.sentiment}</span>
          <span class="conf"><span class="conf-bar"><i style="width:${Math.round(r.confidence * 100)}%"></i></span>${Math.round(r.confidence * 100)}%</span>
          ${r.stars !== null ? `<span>${r.stars} stars</span>` : ""}
          ${r.expected !== null ? `<span>${r.expected === r.sentiment ? "matches stars" : "differs from stars"}</span>` : ""}
        </div>
      </article>`).join("")
    : `<p class="muted">No reviews match this filter.</p>`;
  $("rowcount").textContent = `Showing ${shown.length.toLocaleString()} of ${rows.length.toLocaleString()} matching reviews`
    + (rows.length > shown.length ? " (first 300 shown; export for all)" : "");
}

/* ---------- Exports ---------- */

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

const csvCell = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;

function exportCSV() {
  const lines = [["review", "sentiment", "confidence", "score", "stars", "matches_stars"].join(",")];
  for (const r of analysis.reviews) {
    const match = r.expected == null ? "" : (r.expected === r.sentiment ? "yes" : "no");
    lines.push([r.text, r.sentiment, r.confidence, r.score, r.stars ?? "", match].map(csvCell).join(","));
  }
  download(`sentimentx-reviews-${stamp()}.csv`, "\uFEFF" + lines.join("\n"), "text/csv;charset=utf-8");
}

function exportJSON() {
  const report = {
    product: "SentimentX - The Sentiment Analysis using Machine Learning",
    generated_at: new Date().toISOString(),
    source_file: fileName,
    model: modelInfo ? (modelInfo.model || MODEL_NAME) : MODEL_NAME,
    model_metrics: modelInfo,
    total_reviews: analysis.total,
    mean_score: analysis.mean_score,
    user_label_accuracy_percent: analysis.user_accuracy,
    star_agreement_percent: analysis.star_agreement,
    distribution: analysis.distribution,
    top_words: analysis.top_words,
    issues: analysis.issues,
    reviews: analysis.reviews,
  };
  download(`sentimentx-report-${stamp()}.json`, JSON.stringify(report, null, 2), "application/json");
}

async function exportXLSX() {
  const btn = $("dl-xlsx");
  btn.disabled = true;
  $("exportStatus").textContent = "Building the Excel workbook…";
  try {
    const res = await fetch("/api/export/xlsx", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ file_name: fileName, analysis }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `Export failed (${res.status})`);
    }
    const disposition = res.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([^"]+)"/);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = match ? match[1] : `sentimentx-analysis-${stamp()}.xlsx`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    $("exportStatus").textContent = "";
    toast("Excel workbook downloaded");
  } catch (err) {
    $("exportStatus").textContent = "";
    toast(`${err.message}. The Excel export needs the SentimentX server.`, true);
  } finally {
    btn.disabled = false;
  }
}

/* ---------- Wiring ---------- */

function reset() {
  analysis = null;
  fileName = "";
  filter = "all";
  $("file").value = "";
  $("search").value = "";
  setStatus("");
  setNavEnabled(false);
  $$(".filter").forEach((b) => b.classList.toggle("active", b.dataset.f === "all"));
  $("topIssueCard")?.remove();
  showView("upload");
}

$$("[data-nav] [data-view]").forEach((b) => b.addEventListener("click", () => {
  if (!b.disabled) showView(b.dataset.view);
}));
$("newTop").onclick = reset;
$("newSide").onclick = reset;

$("pick").onclick = () => $("file").click();
$("file").onchange = (e) => e.target.files[0] && analyze(e.target.files[0]);
$("sample").onclick = async () => {
  const blob = await (await fetch("sample_reviews.csv")).blob();
  analyze(new File([blob], "sample_reviews.csv", { type: "text/csv" }));
};

const drop = $("drop");
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) analyze(f); });

$$(".filter").forEach((b) => b.addEventListener("click", () => {
  filter = b.dataset.f;
  $$(".filter").forEach((x) => x.classList.toggle("active", x === b));
  renderReviews();
}));
$("search").addEventListener("input", renderReviews);

$("dl-xlsx").onclick = exportXLSX;
$("dl-csv").onclick = exportCSV;
$("dl-json").onclick = exportJSON;

showView("upload");
loadModel();
