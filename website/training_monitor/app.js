/* Self-contained SVG charts: no CDN, framework, or browser-side SSH access. */
'use strict';
const $ = (selector) => document.querySelector(selector);
const state = {data: null, metric: 'loss', smooth: .7, range: 'all', logarithmic: false, hidden: new Set()};
const metrics = {
  loss: ['Loss 对比', '按真实训练 step 对齐 · 训练 loss，不是验证集指标'],
  grdn: ['梯度范数', '记录的是裁剪前梯度范数 · 训练启用了 grad clip 10'],
  updt_s: ['更新时间', '每个训练 step 的模型更新时间 · 单位：秒'],
  data_s: ['数据等待', '每个训练 step 的数据加载等待时间 · 单位：秒'],
};
const statuses = {running: '训练中', queued: '排队等待双卡', completed: '已完成', failed: '训练异常', stopped: '进程已停止', waiting: '等待数据'};
const escapeHTML = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[char]));
const number = (value, digits = 3) => Number.isFinite(value) ? value.toFixed(digits) : '—';
const scientific = (value) => Number.isFinite(value) ? value.toExponential(1) : '—';
const count = (value) => Number.isFinite(value) ? value.toLocaleString('en-US') : '—';
function timeLabel(seconds) {
  if (!Number.isFinite(seconds)) return '—';
  if (seconds < 60) return `${Math.round(seconds)} 秒`;
  const hours = Math.floor(seconds / 3600), minutes = Math.floor(seconds % 3600 / 60);
  return hours ? `${hours} 小时 ${minutes} 分` : `${minutes} 分`;
}
function ema(values, alpha) {
  let previous;
  return values.map((point) => {
    previous = previous === undefined ? point.y : previous * alpha + point.y * (1 - alpha);
    return {...point, y: previous};
  });
}

function renderRuns(data) {
  $('#runs').innerHTML = data.runs.map((run) => {
    const progress = Math.min(100, 100 * run.step / run.total);
    const latest = run.latest;
    const stale = !data.connected && run.status === 'running';
    return `<article class="run" style="--run-color:${run.color}">
      <div class="run-top"><div><div class="run-title">${escapeHTML(run.label)}</div><div class="run-input">${escapeHTML(run.inputs)} · GPU ${run.gpus.join(' / ')}</div></div><span class="badge ${stale ? 'waiting' : run.status}">${stale ? '数据待同步' : statuses[run.status]}</span></div>
      <div class="run-metrics"><div><small>最新 Loss</small><strong>${number(latest.loss)}</strong></div><div><small>当前学习率</small><strong class="time">${scientific(latest.lr)}</strong></div><div><small>${run.status === 'queued' ? '队列状态' : run.status === 'completed' ? '训练耗时' : '预计剩余'}</small><strong class="time">${run.status === 'queued' ? '等待双卡' : timeLabel(run.status === 'completed' ? run.elapsed : run.eta_seconds)}</strong></div></div>
      <div class="run-progress-meta"><span><b>${count(run.step)}</b> / ${count(run.total)} steps</span><b>${progress.toFixed(1)}%</b></div><div class="progress-track"><span style="width:${progress}%"></span></div>
      <div class="run-bottom"><span>已训练 ${timeLabel(run.elapsed)} · ${number(run.rate, 2)} steps/s</span><span>${run.status === 'queued' ? '前一任务完成后自动启动' : run.status === 'completed' ? '已完成并保存' : run.stop_control?.status === 'saved_and_stopped' ? '已保存并停止' : run.stop_control?.target_step ? `保存 ${count(run.stop_control.target_step)} 后自动停止` : `${run.checkpoints.length} checkpoints`}</span></div>
    </article>`;
  }).join('');
  $('#details-content').innerHTML = data.runs.map((run) => `<div><p><b>${escapeHTML(run.label)}</b> · 输入 ${escapeHTML(run.inputs)} · global batch ${run.batch} · 每卡 ${run.batch / run.gpus.length}<br>输出目录：<code>${escapeHTML(run.output)}</code><br>日志：<code>${escapeHTML(run.log)}</code><br>已保存 checkpoint：${escapeHTML(run.checkpoints.join(' / ') || '尚未保存（每 10,000 步保存）')}</p></div>`).join('') + '<p>架构：ResNet18 / Transformer 768 · 8 heads · encoder 4 层 · decoder 1 层 · chunk 16，执行 4 步。AdamW 3e-5，warmup 1,000，cosine 至 3e-6。相机 1280×720，数据按帧均匀采样。</p>';
}

function renderGPUs(data) {
  const active = data.runs.filter((run) => run.status === 'running');
  const assigned = new Map((active.length ? active : data.runs).flatMap((run) => run.gpus.map((index) => [index, run])));
  const gpus = data.gpus.filter((gpu) => assigned.has(gpu.index));
  $('#gpus').innerHTML = gpus.length ? gpus.map((gpu) => {
    const run = assigned.get(gpu.index);
    return `<div class="gpu-row"><strong title="${escapeHTML(run.label)}">GPU ${gpu.index}</strong><div><div class="gpu-load"><span style="width:${gpu.utilization}%;background:${run.color}"></span></div></div><div class="gpu-memory"><span class="gpu-value">${gpu.utilization}%</span> · ${(gpu.used_mb / 1024).toFixed(1)} GB</div></div>`;
  }).join('') : '等待 GPU 数据…';
}

function axisLabel(value, metric) {
  if (metric === 'lr') return scientific(value);
  if (Math.abs(value) >= 1000) return `${number(value / 1000, 1)}k`;
  if (Math.abs(value) >= 10) return number(value, 1);
  return number(value, metric === 'loss' || metric.endsWith('_s') ? 3 : 2);
}

function chart(svg, metric, small = false) {
  const width = small ? 700 : 1100, height = small ? 235 : 340;
  const margin = {left: small ? 65 : 63, right: 20, top: 16, bottom: 35};
  const innerW = width - margin.left - margin.right, innerH = height - margin.top - margin.bottom;
  const log = !small && state.logarithmic;
  const allRuns = state.data.runs;
  const maxRecordedStep = Math.max(1, ...allRuns.map((run) => run.step));
  const xmin = state.range === 'all' ? 0 : Math.max(0, maxRecordedStep - Number(state.range));
  const series = allRuns.filter((run) => !state.hidden.has(run.id)).map((run) => {
    const raw = run.points.filter((point) => Number.isFinite(point[metric]) && (!log || point[metric] > 0)).map((point) => ({x: point.step, y: point[metric]}));
    // Smooth the complete history before selecting a window to avoid a
    // different EMA value each time the visible window moves forward.
    return {run, raw: raw.filter((p) => p.x >= xmin), smooth: (small ? raw : ema(raw, state.smooth)).filter((p) => p.x >= xmin)};
  });
  const values = series.flatMap((series) => series.raw.map((point) => log ? Math.log10(point.y) : point.y));
  const xmax = Math.max(xmin + 1, maxRecordedStep);
  let ymin = values.length ? Math.min(...values) : 0, ymax = values.length ? Math.max(...values) : 1;
  if (!log) ymin = Math.min(0, ymin);
  const padding = Math.max((ymax - ymin) * .1, ymax === 0 ? 1 : Math.abs(ymax) * .03);
  ymax += padding;
  if (log) ymin -= padding;
  if (ymin === ymax) ymax = ymin + 1;
  const X = (x) => margin.left + (x - xmin) / (xmax - xmin) * innerW;
  const Y = (y) => margin.top + (ymax - (log ? Math.log10(y) : y)) / (ymax - ymin) * innerH;
  const parts = [];
  for (let i = 0; i <= 4; i++) {
    const value = ymin + (ymax - ymin) * i / 4;
    const y = margin.top + innerH * (1 - i / 4);
    parts.push(`<line class="grid" x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}"/><text x="${margin.left - 12}" y="${y + 4}" text-anchor="end">${axisLabel(log ? 10 ** value : value, metric)}</text>`);
  }
  for (let i = 0; i <= 5; i++) {
    const value = xmin + (xmax - xmin) * i / 5;
    parts.push(`<text x="${X(value)}" y="${height - 10}" text-anchor="middle">${count(Math.round(value))}</text>`);
  }
  const path = (points) => points.map((point, index) => `${index ? 'L' : 'M'}${X(point.x).toFixed(2)},${Y(point.y).toFixed(2)}`).join(' ');
  for (const line of series) {
    if (!small) parts.push(`<path d="${path(line.raw)}" fill="none" stroke="${line.run.color}" opacity=".18" stroke-width="1.3"/>`);
    parts.push(`<path d="${path(line.smooth)}" fill="none" stroke="${line.run.color}" stroke-width="${small ? 2 : 2.4}" stroke-linejoin="round" stroke-linecap="round"/>`);
    const last = line.smooth.at(-1);
    if (last) parts.push(`<circle cx="${X(last.x)}" cy="${Y(last.y)}" r="3" fill="${line.run.color}" stroke="white" stroke-width="1.5"/>`);
  }
  if (!values.length) parts.push(`<text x="${width / 2}" y="${height / 2}" text-anchor="middle">${state.hidden.size === allRuns.length ? '点击图例以显示曲线' : '等待所选范围内的训练记录…'}</text>`);
  parts.push(`<line class="chart-axis" x1="${margin.left}" x2="${width - margin.right}" y1="${margin.top + innerH}" y2="${margin.top + innerH}"/>`);
  if (!small) parts.push(`<line id="crosshair" x1="0" x2="0" y1="${margin.top}" y2="${margin.top + innerH}" stroke="#9aa6b8" stroke-dasharray="4 4" visibility="hidden"/>`);
  svg.innerHTML = parts.join('');
  if (!small) {
    $('#point-count').textContent = `${series.reduce((sum, line) => sum + line.raw.length, 0)} records`;
    svg.onpointermove = (event) => {
      const rect = svg.getBoundingClientRect();
      // Use the SVG screen transform to account for preserveAspectRatio on mobile.
      const point = svg.createSVGPoint(); point.x = event.clientX; point.y = event.clientY;
      const cursor = point.matrixTransform(svg.getScreenCTM().inverse());
      if (cursor.x < margin.left || cursor.x > width - margin.right) return hideTooltip();
      const target = xmin + (cursor.x - margin.left) / innerW * (xmax - xmin);
      const nearby = series.map((line) => {
        const closest = line.raw.reduce((best, p) => !best || Math.abs(p.x - target) < Math.abs(best.x - target) ? p : best, null);
        return closest ? {run: line.run, point: closest} : null;
      }).filter(Boolean);
      if (!nearby.length) return hideTooltip();
      $('#crosshair').setAttribute('x1', cursor.x); $('#crosshair').setAttribute('x2', cursor.x); $('#crosshair').setAttribute('visibility', 'visible');
      const tooltip = $('#tooltip');
      tooltip.innerHTML = `<b>Step ≈ ${count(Math.round(target))} · 原始值</b>` + nearby.map(({run, point}) => `<div class="tooltip-row"><span style="color:${run.color}">${escapeHTML(run.label)}</span><span>${axisLabel(point.y, metric)} <small>@ ${point.x}</small></span></div>`).join('');
      tooltip.hidden = false;
      tooltip.style.left = `${Math.max(0, Math.min(event.clientX - rect.left + 14, rect.width - 260))}px`;
      tooltip.style.top = `${Math.max(0, event.clientY - rect.top - 70)}px`;
    };
    svg.onpointerleave = hideTooltip;
  }
}
function hideTooltip() {
  $('#tooltip').hidden = true;
  $('#crosshair')?.setAttribute('visibility', 'hidden');
}
function renderCharts() {
  if (!state.data) return;
  hideTooltip();
  $('#chart-title').textContent = metrics[state.metric][0];
  $('#chart-description').textContent = metrics[state.metric][1];
  chart($('#main-chart'), state.metric);
  chart($('#lr-chart'), 'lr', true);
}
function render(data) {
  state.data = data;
  $('#target-steps').textContent = count(Math.max(...data.runs.map((run) => run.total)));
  $('#connection').className = `connection ${data.connected ? 'live' : 'offline'}`;
  $('#connection').textContent = data.connected ? '实时同步' : '连接待恢复';
  $('#updated').textContent = data.last_success ? `最近同步 ${new Date(data.last_success * 1000).toLocaleTimeString('zh-CN', {hour12: false})}` : '正在读取服务器…';
  $('#notice').hidden = data.connected;
  $('#notice').textContent = data.error || '等待与服务器建立连接。历史曲线会在首次成功同步后显示。';
  $('#refresh-note').textContent = `每 ${data.poll_seconds} 秒同步服务器 · 浏览器自动刷新`;
  renderRuns(data); renderGPUs(data);
  $('#legend').innerHTML = data.runs.map((run) => `<button class="${state.hidden.has(run.id) ? 'inactive' : ''}" data-run="${run.id}" aria-pressed="${!state.hidden.has(run.id)}"><i style="background:${run.color}"></i>${escapeHTML(run.label)}</button>`).join('');
  $('#legend').querySelectorAll('button').forEach((button) => button.onclick = () => {
    const id = button.dataset.run;
    state.hidden.has(id) ? state.hidden.delete(id) : state.hidden.add(id);
    render(state.data);
  });
  renderCharts();
}
async function refresh() {
  try {
    const response = await fetch('/api/status', {cache: 'no-store', signal: AbortSignal.timeout(8000)});
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    $('#connection').className = 'connection offline'; $('#connection').textContent = '本机服务未连接';
    $('#notice').hidden = false;
    $('#notice').textContent = '暂时无法连接本机监控服务，已保留页面上的历史数据。请确认 training_dashboard.py 仍在运行。';
    // Keep all charts and values, but avoid displaying stale running badges.
    if (state.data) $('#runs').querySelectorAll('.badge').forEach((badge) => { if (badge.textContent === '训练中') {badge.textContent = '数据待同步'; badge.className = 'badge waiting';} });
  } finally {
    window.setTimeout(refresh, 5000);
  }
}
document.querySelectorAll('[data-metric]').forEach((button) => button.onclick = () => {
  state.metric = button.dataset.metric;
  document.querySelectorAll('[data-metric]').forEach((item) => item.classList.toggle('active', item === button));
  renderCharts();
});
$('#smooth').oninput = (event) => { state.smooth = Number(event.target.value) / 100; $('#smooth-value').textContent = state.smooth.toFixed(2); renderCharts(); };
$('#range').onchange = (event) => { state.range = event.target.value; renderCharts(); };
$('#log-scale').onchange = (event) => { state.logarithmic = event.target.checked; renderCharts(); };
refresh();
