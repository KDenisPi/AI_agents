"use strict";

// How often to ask /api/poll whether the pending request is done.
const POLL_INTERVAL_MS = 750;
// Client-side ceiling on a pending request, independent of whatever the
// server itself is willing to wait on a model call - voice synthesis
// roughly doubles response time, so this stays generous.
const CLIENT_TIMEOUT_MS = 180000;

const statusEl = document.getElementById("status");

const btnCurrent = document.getElementById("btn-current");
const btnBattery = document.getElementById("btn-battery");
const btnLastHours = document.getElementById("btn-last-hours");
const hoursSelect = document.getElementById("hours-select");
const btnRange = document.getElementById("btn-range");
const fromDateInput = document.getElementById("from-date");
const toDateInput = document.getElementById("to-date");
const audioToggle = document.getElementById("toggle-audio");
const graphToggle = document.getElementById("toggle-graph");

const metricPanel = document.getElementById("metric-panel");
const metricSelect = document.getElementById("metric-select");

const answerStatus = document.getElementById("answer-status");

const graphArea = document.getElementById("graph-area");
const graphImage = document.getElementById("graph-image");
const equalizer = document.getElementById("equalizer");
const answerStrip = document.getElementById("answer-strip");
const audioPlayer = document.getElementById("audio-player");

const lockable = [
  btnCurrent, btnBattery, btnLastHours, hoursSelect, btnRange,
  fromDateInput, toDateInput, audioToggle, graphToggle,
];

let locked = false;
let requestStartedAt = 0;
let pollTimer = null;
let elapsedTimer = null;
let timeoutTimer = null;
let graphsForCurrentResult = {};

function init() {
  for (let h = 1; h <= 48; h++) {
    const opt = document.createElement("option");
    opt.value = String(h);
    opt.textContent = `${h}h`;
    hoursSelect.appendChild(opt);
  }
  hoursSelect.value = "12";

  const today = new Date();
  const weekAgo = new Date(today.getTime() - 7 * 86400000);
  toDateInput.value = isoDateLocal(today);
  fromDateInput.value = isoDateLocal(weekAgo);

  btnCurrent.addEventListener("click", requestCurrent);
  btnBattery.addEventListener("click", requestBattery);
  btnLastHours.addEventListener("click", requestLastHours);
  btnRange.addEventListener("click", requestRange);
  metricSelect.addEventListener("change", () => showMetric(metricSelect.value));
}

function isoDateLocal(d) {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function titleCase(s) {
  return s.length ? s[0].toUpperCase() + s.slice(1) : s;
}

// ---- Request id + URL building ----

function buildRequestId() {
  return `web-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

function buildUrl(path, params) {
  const usp = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === false) continue;
    usp.set(k, v === true ? "" : String(v));
  }
  const qs = usp.toString();
  return qs ? `${path}?${qs}` : path;
}

// ---- The four requests the sidebar can start ----

function requestCurrent() {
  if (locked) return;
  const requestId = buildRequestId();
  const params = { request_id: requestId };
  if (audioToggle.checked) params.voice = true;
  startRequest(buildUrl("/api/current", params), requestId);
}

function requestBattery() {
  if (locked) return;
  const requestId = buildRequestId();
  const params = { request_id: requestId };
  if (audioToggle.checked) params.voice = true;
  startRequest(buildUrl("/api/current_battery", params), requestId);
}

function requestLastHours() {
  if (locked) return;
  const requestId = buildRequestId();
  const hours = parseInt(hoursSelect.value, 10);
  const params = { request_id: requestId, hours };
  if (graphToggle.checked) params.graph = true;
  // No voice param here - /api/outside_last_hours has no voice option.
  startRequest(buildUrl("/api/outside_last_hours", params), requestId);
}

function requestRange() {
  if (locked) return;
  const fromVal = fromDateInput.value;
  const toVal = toDateInput.value;
  if (!fromVal || !toVal) {
    showImmediateError("Error: choose both a From and To date.");
    return;
  }
  const hours = hoursForRange(fromVal, toVal);
  if (hours === null) {
    showImmediateError("Error: 'From' date must be on or before 'To' date.");
    return;
  }
  const requestId = buildRequestId();
  const params = { request_id: requestId, hours };
  if (graphToggle.checked) params.graph = true;
  startRequest(buildUrl("/api/outside_last_hours", params), requestId);
}

// The server has no dedicated from/to endpoint - only /api/outside_last_hours
// (hours before now). Whole-day span, inclusive of both endpoints, converted
// to hours so "Run range query" can reuse that endpoint.
function hoursForRange(fromStr, toStr) {
  const from = new Date(`${fromStr}T00:00:00`);
  const to = new Date(`${toStr}T00:00:00`);
  if (to < from) return null;
  const days = Math.round((to - from) / 86400000) + 1;
  return days * 24;
}

// ---- Request lifecycle: send, poll, resolve ----

function startRequest(url, requestId) {
  setLocked(true);
  clearResults();
  requestStartedAt = Date.now();
  setAnswerStatusThinking();
  timeoutTimer = setTimeout(onClientTimeout, CLIENT_TIMEOUT_MS);

  fetch(url)
    .then(async (response) => {
      if (!response.ok) {
        const detail = await safeErrorDetail(response);
        finishWithError(`Error ${response.status}: ${detail}`);
        return;
      }
      beginPolling(requestId);
    })
    .catch((err) => {
      finishWithError(`Error: ${err.message || "could not reach server"}`);
    });
}

async function safeErrorDetail(response) {
  try {
    const body = await response.json();
    return body.error || response.statusText || "request failed";
  } catch {
    return response.statusText || "request failed";
  }
}

function beginPolling(requestId) {
  pollTimer = setInterval(() => pollOnce(requestId), POLL_INTERVAL_MS);
  pollOnce(requestId);
}

async function pollOnce(requestId) {
  let response;
  try {
    response = await fetch(buildUrl("/api/poll", { request_id: requestId }));
  } catch {
    // A single flaky poll on the LAN isn't fatal - the client-side timeout
    // is what gives up for real.
    return;
  }

  if (response.status === 404) {
    finishWithError("Error 404: server lost track of this request.");
    return;
  }
  if (!response.ok) {
    const detail = await safeErrorDetail(response);
    finishWithError(`Error ${response.status}: ${detail}`);
    return;
  }

  const body = await response.json();
  if (body.pending) return;
  if (body.error) {
    finishWithError(`Error: ${body.error}`);
    return;
  }
  finishWithResult(body);
}

function onClientTimeout() {
  finishWithError(`Error: timed out after ${Math.round(CLIENT_TIMEOUT_MS / 1000)}s waiting for a response.`);
}

function finishWithError(message) {
  stopPollingAndTimers();
  setLocked(false);
  setAnswerStatusIdle();
  showAnswerText(message, true);
}

function finishWithResult(payload) {
  stopPollingAndTimers();
  setLocked(false);
  setAnswerStatusIdle();
  showAnswerText(payload.text || "", false);
  applyAudio(payload);
  applyGraphs(payload);
}

function stopPollingAndTimers() {
  clearInterval(pollTimer);
  clearTimeout(timeoutTimer);
  clearInterval(elapsedTimer);
  pollTimer = null;
  timeoutTimer = null;
  elapsedTimer = null;
}

// ---- UI state ----

function setLocked(value) {
  locked = value;
  lockable.forEach((el) => { el.disabled = value; });
  statusEl.textContent = value ? "Busy" : "Ready";
  statusEl.classList.toggle("busy", value);
}

function setAnswerStatusThinking() {
  answerStatus.classList.remove("idle");
  answerStatus.classList.add("thinking");
  updateElapsed();
  elapsedTimer = setInterval(updateElapsed, 1000);
}

function updateElapsed() {
  const elapsed = Math.floor((Date.now() - requestStartedAt) / 1000);
  const m = Math.floor(elapsed / 60);
  const s = String(elapsed % 60).padStart(2, "0");
  answerStatus.textContent = `Thinking... ${m}:${s}`;
}

function setAnswerStatusIdle() {
  answerStatus.classList.remove("thinking");
  answerStatus.classList.add("idle");
  answerStatus.textContent = "Answer text";
}

function showAnswerText(text, isError) {
  answerStrip.textContent = text;
  answerStrip.classList.toggle("error", !!isError);
}

function showImmediateError(message) {
  clearResults();
  showAnswerText(message, true);
}

function clearResults() {
  showAnswerText("", false);
  graphImage.removeAttribute("src");
  graphImage.hidden = true;
  equalizer.hidden = true;
  graphArea.hidden = true;
  metricPanel.hidden = true;
  metricSelect.innerHTML = "";
  graphsForCurrentResult = {};
  audioPlayer.pause();
  audioPlayer.removeAttribute("src");
}

// ---- Applying a finished result ----

function applyAudio(payload) {
  if (!payload.audio_url) return;
  audioPlayer.src = payload.audio_url;
  audioPlayer.onended = () => { equalizer.hidden = true; };
  audioPlayer.play().catch((err) => {
    console.warn("Audio playback blocked by the browser:", err);
  });
}

function applyGraphs(payload) {
  const graphs = payload.graphs || {};
  const metricNames = Object.keys(graphs);
  graphsForCurrentResult = graphs;

  const graphWanted = graphToggle.checked && metricNames.length > 0;
  const audioWanted = audioToggle.checked && !!payload.audio_url;

  if (graphWanted) {
    graphArea.hidden = false;
    graphImage.hidden = false;
    equalizer.hidden = true;
    if (metricNames.length > 1) {
      populateMetricSelect(metricNames);
      metricPanel.hidden = false;
    } else {
      metricPanel.hidden = true;
    }
    showMetric(metricNames[0]);
  } else if (audioWanted) {
    graphArea.hidden = false;
    graphImage.hidden = true;
    equalizer.hidden = false;
    metricPanel.hidden = true;
  } else {
    graphArea.hidden = true;
    metricPanel.hidden = true;
  }
}

function populateMetricSelect(names) {
  metricSelect.innerHTML = "";
  for (const name of names) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = titleCase(name);
    metricSelect.appendChild(opt);
  }
  metricSelect.value = names[0];
}

function showMetric(name) {
  const url = graphsForCurrentResult[name];
  if (!url) return;
  graphImage.src = url;
  graphImage.alt = titleCase(name);
}

init();
