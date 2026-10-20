// Render and exercise chart controls without external JS dependencies.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');

class Element {
  constructor() {
    this.innerHTML = ''; this.textContent = ''; this.style = {}; this.dataset = {};
    this.attributes = {}; this.classList = {toggle() {}};
  }
  querySelectorAll() {return [];}
  setAttribute(key, value) {this.attributes[key] = value;}
  getBoundingClientRect() {return {left: 0, top: 0, width: 1100, height: 340};}
  createSVGPoint() {return {matrixTransform() {return this;}};}
  getScreenCTM() {return {inverse() {return {};}};}
}
const elements = new Map();
const document = {
  querySelector(selector) {if (!elements.has(selector)) elements.set(selector, new Element()); return elements.get(selector);},
  querySelectorAll() {return [];},
};
const context = vm.createContext({document, window: {setTimeout() {}}, console, AbortSignal});
const source = fs.readFileSync(path.join(__dirname, '../website/training_monitor/app.js'), 'utf8');
vm.runInContext(source.replace(/refresh\(\);\s*$/, ''), context);
const sample = {
  connected: true, last_success: Date.now() / 1000, poll_seconds: 10, gpus: [],
  runs: ['camera', 'z'].map((id, index) => ({id, label: index ? 'Camera + Z' : 'Camera only',
    color: index ? '#12a68b' : '#3975f6', inputs: 'RGB', gpus: [index], frames: 71746,
    episodes: 745, batch: 16, output: '/output', log: '/log', status: 'running',
    step: 1660, total: 30000, elapsed: 450, eta_seconds: 5000, rate: 4,
    checkpoints: [], latest: {loss: .2, lr: 3e-5}, points: [
      {step: 1600, loss: .4, lr: 2e-5, grdn: 0, data_s: .1, updt_s: .12},
      {step: 1640, loss: .2, lr: 3e-5, grdn: 5, data_s: 0, updt_s: .14},
      {step: 1660, loss: .1, lr: 3e-5, grdn: 4, data_s: .2, updt_s: .13},
    ]})),
};
context.sample = sample;
vm.runInContext('render(sample)', context);
assert.match(document.querySelector('#runs').innerHTML, /Camera only/);
assert.match(document.querySelector('#main-chart').innerHTML, /<path/);
assert.match(document.querySelector('#lr-chart').innerHTML, /<path/);
assert.equal(vm.runInContext('ema([{y:1},{y:0}], .5)[1].y', context), .5);
for (const metric of ['loss', 'grdn', 'updt_s', 'data_s']) {
  vm.runInContext(`state.metric = '${metric}'; state.logarithmic = true; renderCharts()`, context);
  assert.doesNotMatch(document.querySelector('#main-chart').innerHTML, /NaN|Infinity/);
}
document.querySelector('#smooth').oninput({target: {value: '85'}});
assert.equal(document.querySelector('#smooth-value').textContent, '0.85');
document.querySelector('#range').onchange({target: {value: '2000'}});
assert.doesNotMatch(document.querySelector('#main-chart').innerHTML, /NaN|Infinity/);
document.querySelector('#main-chart').onpointermove({clientX: 500, clientY: 120});
assert.match(document.querySelector('#tooltip').innerHTML, /原始值/);
vm.runInContext('state.hidden = new Set(sample.runs.map(run => run.id)); renderCharts()', context);
assert.match(document.querySelector('#main-chart').innerHTML, /点击图例/);
sample.connected = false; sample.error = 'SSH 认证已失效';
vm.runInContext('render(sample)', context);
assert.match(document.querySelector('#runs').innerHTML, /数据待同步/);
assert.equal(document.querySelector('#notice').hidden, false);
assert.match(document.querySelector('#notice').textContent, /SSH/);
console.log('PASS: actual chart rendering, EMA, four metrics, log axis, range, tooltip, hidden series and offline state');
