const DATA = {
  models: 'assets/results/model_comparison.csv',
  robustness: 'assets/results/sensor_robustness.csv',
  orderedHoldout: 'assets/results/ordered_holdout_comparison.csv',
  barnEvaluation: 'assets/results/barn_environment_generalization.csv',
  turtlebotDropout: 'assets/results/turtlebot_recorded_dropout.csv',
  realGoalValidation: 'assets/results/real_goal_run_validation.csv',
  realRunQuality: 'assets/results/real_run_sensor_quality.csv',
  trials: 'assets/results/navigation_trial_steps.csv?v=unique-start-goals-2026-11-17',
  trialSummary: 'assets/results/navigation_trials.csv?v=unique-start-goals-2026-11-17',
  browserModels: 'assets/models/goal_navigation_models.json'
};

const modelNotes = {
  'Dummy': 'Most-frequent-class baseline; ignores the sensor features.',
  'Logistic Regression': 'Linear classifier with median imputation and standard-scaled readings.',
  'RBF SVM': 'RBF-kernel classifier with median imputation and standard-scaled readings.',
  'K-Nearest Neighbors': 'Five-neighbor uniform vote with median imputation and standard-scaled readings.',
  'Random Forest': '300-tree ensemble with median imputation; numeric readings are not scaled.',
  'Gradient Boosting': '100 boosting stages, learning rate 0.1, depth 3, and median imputation.'
};

const modelDiscussionNotes = {
  'Dummy': {
    reading: 'This is the majority-class reference. A low macro-F1 beside its accuracy shows that predicting the common labels can look acceptable while missing minority actions.',
    improve: 'Keep it as the floor for every comparison. Do not use it to move the robot; collect more examples for minority actions before expecting their estimates to be stable.'
  },
  'Logistic Regression': {
    reading: 'The linear model is fast and easy to inspect, but its lower macro-F1 suggests a straight decision boundary misses sensor-action relationships captured by nonlinear models.',
    improve: 'Tune regularization within training-fold CV and inspect class-wise errors. Keep it as an interpretable reference; compare any added nonlinear features on the same held-out split.'
  },
  'RBF SVM': {
    reading: 'The RBF kernel captures nonlinear boundaries and improves on the linear reference, but it remains behind the two tree ensembles on this split.',
    improve: 'Select C and gamma using training-only cross-validation, then inspect minority-action recall and probability calibration. Compare any class weighting using the same folds.'
  },
  'K-Nearest Neighbors': {
    reading: 'Neighbors provide a local, instance-based decision and score above the baseline, but performance trails SVM and the ensembles. Its prediction cost also grows with stored training examples.',
    improve: 'Tune neighbor count, distance weighting, and metric inside cross-validation. Evaluate inference time as the training set grows and check minority-class recall.'
  },
  'Random Forest': {
    reading: 'Random Forest is close to Gradient Boosting on this holdout and had a shorter fitting time in this run. It is a strong alternative when training cost matters.',
    improve: 'Tune tree count, depth, and feature sampling inside training-only folds. Inspect feature importance for sensor cues, and treat the reported timing as machine-specific.'
  },
  'Gradient Boosting': {
    reading: 'Gradient Boosting is the strongest candidate in this comparison, but its remaining mistakes show a practical risk: a right-turn situation can be classified as forward.',
    improve: 'Treat it as the strongest candidate for this dataset, not a safe controller. Inspect class probabilities and consider an abstain rule for uncertain readings; then evaluate the controller separately for goal completion and safety.'
  }
};

const classInfo = [
  { label: '1 · Forward', count: 2205, pct: 40.41 },
  { label: '2 · Slight right', count: 2097, pct: 38.43 },
  { label: '3 · Sharp right', count: 328, pct: 6.01 },
  { label: '4 · Slight left', count: 826, pct: 15.14 }
];

function parseCsv(text) {
  const rows = [];
  let row = [], field = '', quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (quoted && char === '"' && text[i + 1] === '"') { field += '"'; i += 1; }
    else if (char === '"') quoted = !quoted;
    else if (char === ',' && !quoted) { row.push(field); field = ''; }
    else if ((char === '\n' || char === '\r') && !quoted) {
      if (char === '\r' && text[i + 1] === '\n') i += 1;
      row.push(field); field = '';
      if (row.some((cell) => cell !== '')) rows.push(row);
      row = [];
    } else field += char;
  }
  if (field || row.length) { row.push(field); rows.push(row); }
  const [header, ...data] = rows;
  return data.map((values) => Object.fromEntries(header.map((key, index) => [key, values[index] ?? ''])));
}

async function csv(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not load ${path}`);
  return parseCsv(await response.text());
}

async function loadRealRunValidation() {
  const state=document.querySelector('#real-run-result-state');
  try {
    const rows=await csv(DATA.realGoalValidation);
    if(!rows.length) throw new Error('The grouped run report has no model rows yet');
    const ordered=[...rows].sort((a,b)=>Number(b.macro_f1_mean)-Number(a.macro_f1_mean));
    const colors={'Dummy':'#91a0ac','Logistic Regression':'#20a88e','RBF SVM':'#e5ae54','K-Nearest Neighbors':'#f07868','Random Forest':'#639ef0','Gradient Boosting':'#a978df'};
    document.querySelector('#real-run-body').innerHTML=ordered.map(r=>`<tr><th scope="row">${r.model}</th><td>${Number(r.macro_f1_mean).toFixed(3)}</td><td>${Number(r.macro_f1_std).toFixed(3)}</td><td>${Number(r.accuracy_mean).toFixed(3)}</td><td>${r.group_folds}</td></tr>`).join('');
    const svg=document.querySelector('#real-run-chart'), width=720,left=185,right=50,top=26,rowH=38,bottom=30,height=top+ordered.length*rowH+bottom,plotW=width-left-right;
    const grid=[0,.25,.5,.75,1].map(v=>{const x=left+v*plotW;return `<line class="axis-line" x1="${x}" x2="${x}" y1="${top-8}" y2="${top+ordered.length*rowH-7}"/><text class="axis-label" x="${x}" y="${height-6}" text-anchor="middle">${v.toFixed(2)}</text>`}).join('');
    const bars=ordered.map((r,i)=>{const y=top+i*rowH,mean=Number(r.macro_f1_mean),sd=Number(r.macro_f1_std),color=colors[r.model]||'#536779',x0=left+Math.max(0,mean-sd)*plotW,x1=left+Math.min(1,mean+sd)*plotW;return `<text class="axis-label" x="${left-10}" y="${y+15}" text-anchor="end">${r.model}</text><rect x="${left}" y="${y}" width="${plotW*Math.max(0,Math.min(1,mean))}" height="20" rx="3" fill="${color}"><title>${r.model}: mean macro-F1 ${mean.toFixed(3)} ± ${sd.toFixed(3)} over ${r.group_folds} grouped folds</title></rect><line x1="${x0}" x2="${x1}" y1="${y+10}" y2="${y+10}" stroke="#142536" stroke-width="2"/><text class="axis-label" x="${Math.min(width-right,left+plotW*mean+7)}" y="${y+14}">${mean.toFixed(3)}</text>`}).join('');
    svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
    svg.innerHTML=`${grid}${bars}<text class="axis-label" x="${left}" y="12">GROUPED HELD-OUT-RUN MACRO-F1 · MEAN ± SD</text>`;
    document.querySelector('#real-run-results').hidden=false;
    state.textContent=`Loaded results from ${ordered[0].runs_available} independent robot runs. All rows from each run stay in one fold.`;
  } catch (error) {
    state.textContent='Awaiting new independent robot runs. Record multiple bags, then run the extraction and grouped-evaluation commands above; the chart and downloadable results will appear here.';
  }
}

async function loadPhysicalRunQuality() {
  const state=document.querySelector('#real-fault-result-state');
  try {
    const rows=await csv(DATA.realRunQuality);
    if(!rows.length) return;
    document.querySelector('#real-fault-body').innerHTML=rows.map(r=>`<tr><th scope="row">${r.run_id}</th><td>${r.fault_condition}</td><td>${Number(r.scan_messages)}</td><td>${Number(r.median_scan_period_s).toFixed(4)}</td><td>${Number(r.scan_gaps_over_3x_median)}</td><td>${Number(r.aligned_action_rows)}</td></tr>`).join('');
    document.querySelector('#real-fault-results').hidden=false;
    document.querySelector('#real-fault-download').hidden=false;
    state.textContent=`Loaded ${rows.length} run summaries. Conditions are operator annotations; scan gaps are diagnostic signals, not proof of hardware faults.`;
  } catch (error) {
    state.textContent='After recording and extracting runs, this table will list scan coverage and long gaps by run. A gap alone cannot prove a hardware fault.';
  }
}

function renderClassBars() {
  const host = document.querySelector('#class-bars');
  host.innerHTML = classInfo.map((item) => `<div class="class-bar-row"><span class="class-bar-name">${item.label}</span><span class="bar-track"><i class="bar-fill" style="width:${item.pct}%"></i></span><span class="class-bar-value">${item.count.toLocaleString()} · ${item.pct.toFixed(1)}%</span></div>`).join('');
}

let modelRows = [];
const metricLabels = {
  cv_macro_f1_mean: 'CV macro-F1', test_macro_f1: 'holdout macro-F1', test_accuracy: 'holdout accuracy',
  test_balanced_accuracy: 'balanced accuracy', prediction_ms_per_sample: 'inference time'
};

function fmtMetric(value, metric) {
  const number = Number(value);
  return metric === 'prediction_ms_per_sample' ? `${number.toFixed(3)} ms` : number.toFixed(3);
}

function renderModels(metric = 'cv_macro_f1_mean') {
  const host = document.querySelector('#model-table');
  document.querySelector('#model-table-head').innerHTML = `<span>RANK</span><span>MODEL</span><span>RELATIVE PERFORMANCE</span><span>${metric === 'prediction_ms_per_sample' ? 'MS / SAMPLE' : metricLabels[metric].toUpperCase()}</span>`;
  const lowerBetter = metric === 'prediction_ms_per_sample';
  const sorted = [...modelRows].sort((a, b) => lowerBetter ? Number(a[metric]) - Number(b[metric]) : Number(b[metric]) - Number(a[metric]));
  const bestValue = Number(sorted[0][metric]);
  host.innerHTML = sorted.map((row, index) => {
    const value = Number(row[metric]);
    const width = lowerBetter ? Math.max(3, (bestValue / value) * 100) : Math.max(3, value * 100);
    const key = row.model.replaceAll('"', '&quot;');
    const classes = [index === 0 ? 'best' : '', row.model === 'Dummy' ? 'baseline' : ''].filter(Boolean).join(' ');
    return `<button class="model-row ${classes}" type="button" data-model="${key}" aria-pressed="${index === 0}"><span class="model-rank">${String(index + 1).padStart(2, '0')}</span><span class="model-name">${row.model}${index === 0 ? ' <span aria-label="ranked first">✦</span>' : ''}${row.model === 'Dummy' ? ' <small>BASELINE</small>' : ''}</span><span class="model-bar-track" aria-hidden="true"><i class="model-bar-fill" style="width:${Math.min(100, width)}%"></i></span><span class="model-value">${fmtMetric(value, metric)}</span></button>`;
  }).join('');
  host.querySelectorAll('.model-row').forEach((button) => button.addEventListener('click', () => {
    host.querySelectorAll('.model-row').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
    showModelDetail(button.dataset.model, metric);
  }));
  showModelDetail(sorted[0].model, metric);
}

function showModelDetail(name, metric) {
  const row = modelRows.find((item) => item.model === name);
  if (!row) return;
  const detail = document.querySelector('#model-detail');
  detail.innerHTML = `<b>${name}</b> — ${modelNotes[name] || ''} <span>Test macro-F1 ${Number(row.test_macro_f1).toFixed(3)} · accuracy ${Number(row.test_accuracy).toFixed(3)} · CV ${Number(row.cv_macro_f1_mean).toFixed(3)} ± ${Number(row.cv_macro_f1_std).toFixed(3)}</span>`;
  const discussionSelect = document.querySelector('#discussion-model');
  if (discussionSelect) {
    discussionSelect.value = name;
    const note = modelDiscussionNotes[name] || {};
    document.querySelector('#discussion-reading').textContent = note.reading || '';
    document.querySelector('#discussion-improvement').textContent = note.improve || '';
  }
}

let robustnessRows = [];
let orderedHoldoutRows = [];
let barnEvaluationRows = [];
let turtlebotDropoutRows = [];
const robustnessModelColors = {'Dummy':'#91a0ac','Logistic Regression':'#20a88e','RBF SVM':'#e5ae54','K-Nearest Neighbors':'#f07868','Random Forest':'#639ef0','Gradient Boosting':'#a978df'};
function renderRobustnessChart(experiment = 'gaussian_noise') {
  const svg = document.querySelector('#noise-chart');
  const config = {
    gaussian_noise: { title:'GAUSSIAN NOISE · FEATURE SD', x:r=>Number(r.condition_value), labels:['0','0.05','0.10','0.15','0.20'], max:.2 },
    random_dropout: { title:'RANDOM SENSOR DROPOUT · % FEATURES / ROW', x:r=>Number(r.condition_value)*100, labels:['0%','25%','50%','75%'], max:75 },
    available_sensors: { title:'AVAILABLE SENSOR CHANNELS', x:r=>Number(r.condition_value), labels:['6','12','18','24'], max:24 }
  }[experiment];
  const rows=robustnessRows.filter(r=>r.experiment===experiment), models=[...new Set(rows.map(r=>r.model))];
  const width=520,height=280,left=46,right=15,top=20,bottom=50,plotW=width-left-right,plotH=height-top-bottom;
  const x=v=>left+v/config.max*plotW, y=v=>top+(1-v)*plotH;
  const ticks=[0,.25,.5,.75,1];
  const grid=ticks.map(t=>`<line class="axis-line" x1="${left}" x2="${width-right}" y1="${y(t)}" y2="${y(t)}"/><text class="axis-label" x="${left-8}" y="${y(t)+3}" text-anchor="end">${t.toFixed(2)}</text>`).join('');
  const ordered=[...new Set(rows.map(r=>Number(r.condition_value)))].sort((a,b)=>experiment==='available_sensors'?a-b:a-b);
  const xLabels=ordered.map(v=>`<text class="axis-label" x="${x(experiment==='random_dropout'?v*100:v)}" y="${height-24}" text-anchor="middle">${experiment==='random_dropout'?`${Math.round(v*100)}%`:experiment==='gaussian_noise'?v.toFixed(2):v}</text>`).join('');
  const lines=models.map(name=>{
    const color=robustnessModelColors[name]||'#536779';
    const points=ordered.map(value=>{const r=rows.find(a=>a.model===name&&Number(a.condition_value)===value);return r?{x:experiment==='random_dropout'?value*100:value,y:Number(r.macro_f1_mean),sd:Number(r.macro_f1_std)||0}:null}).filter(Boolean);
    const coords=points.map(p=>`${x(p.x)},${y(p.y)}`).join(' ');
    return `<polyline points="${coords}" fill="none" stroke="${color}" stroke-width="2.5"/>${points.map(p=>`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="4" fill="${color}"><title>${name}: macro-F1 ${p.y.toFixed(3)} ± ${p.sd.toFixed(3)} across 10 repeats</title></circle>`).join('')}`;
  }).join('');
  svg.innerHTML=`${grid}${lines}${xLabels}<text class="axis-label" x="${left}" y="12">MACRO-F1 · MEAN ± SD</text><text class="axis-label" x="${width-right}" y="${height-5}" text-anchor="end">${config.title}</text>`;
  const legend=document.querySelector('#robustness-legend');
  legend.innerHTML=models.map(n=>`<span><i style="background:${robustnessModelColors[n]||'#536779'}"></i>${n}</span>`).join('');
  const baseline=rows.filter(r=>Number(r.condition_value)===(experiment==='available_sensors'?24:0));
  const hardest=rows.filter(r=>Number(r.condition_value)===(experiment==='available_sensors'?6:Math.max(...ordered)));
  const mean=a=>a.length?a.reduce((s,r)=>s+Number(r.macro_f1_mean),0)/a.length:0;
  document.querySelector('#noise-score').textContent=`${mean(baseline).toFixed(3)} → ${mean(hardest).toFixed(3)}`;
  document.querySelector('#noise-delta').textContent='mean macro-F1 across six models · 10 repeats per condition';
}

function renderOrderedHoldout() {
  const host=document.querySelector('#ordered-holdout-body');
  host.innerHTML=orderedHoldoutRows.map(row=>{
    const random=modelRows.find(item=>item.model===row.model);
    return `<tr><th scope="row">${row.model}</th><td>${random?Number(random.test_macro_f1).toFixed(3):'—'}</td><td>${Number(row.test_macro_f1).toFixed(3)}</td><td>${Number(row.test_accuracy).toFixed(3)}</td></tr>`;
  }).join('');
}

function renderBarnEvaluation() {
  const rows=[...barnEvaluationRows].sort((a,b)=>Number(b.route_success_rate)-Number(a.route_success_rate));
  document.querySelector('#barn-eval-body').innerHTML=rows.map(r=>`<tr><th scope="row">${r.model}</th><td>${Number(r.heldout_action_macro_f1).toFixed(3)}</td><td>${Number(r.mean_per_map_macro_f1).toFixed(3)}</td><td>${Number(r.route_successes)} / ${Number(r.heldout_maps)}</td><td>${Number(r.collision_failures)}</td></tr>`).join('');
  const svg=document.querySelector('#barn-route-chart'), colors={Dummy:'#91a0ac','Logistic Regression':'#20a88e','RBF SVM':'#e5ae54','K-Nearest Neighbors':'#f07868','Random Forest':'#639ef0','Gradient Boosting':'#a978df'};
  const left=184, right=35, top=26, rowH=38, width=720, bottom=30, height=top+rows.length*rowH+bottom, plotW=width-left-right;
  const grid=[0,.25,.5,.75,1].map(v=>{const x=left+v*plotW;return `<line class="axis-line" x1="${x}" x2="${x}" y1="${top-8}" y2="${top+rows.length*rowH-7}"/><text class="axis-label" x="${x}" y="${height-6}" text-anchor="middle">${Math.round(v*100)}%</text>`}).join('');
  const bars=rows.map((r,i)=>{const y=top+i*rowH, value=Number(r.route_success_rate);return `<text class="axis-label" x="${left-10}" y="${y+15}" text-anchor="end">${r.model}</text><rect x="${left}" y="${y}" width="${plotW*value}" height="20" rx="3" fill="${colors[r.model]||'#536779'}"><title>${r.model}: ${r.route_successes} of ${r.heldout_maps} held-out goals reached</title></rect><text class="axis-label" x="${Math.min(width-right-2,left+plotW*value+7)}" y="${y+14}">${r.route_successes}/${r.heldout_maps}</text>`}).join('');
  svg.setAttribute('viewBox',`0 0 ${width} ${height}`);
  svg.innerHTML=`${grid}${bars}<text class="axis-label" x="${left}" y="12">HELD-OUT ROUTE SUCCESS RATE</text>`;
}

function renderTurtlebotDropout() {
  const names=[...new Set(turtlebotDropoutRows.map(r=>r.model))], rates=[0,.25,.5,.75];
  const colors={Dummy:'#91a0ac','Logistic Regression':'#20a88e','RBF SVM':'#e5ae54','K-Nearest Neighbors':'#f07868','Random Forest':'#639ef0','Gradient Boosting':'#a978df'};
  document.querySelector('#turtlebot-dropout-body').innerHTML=names.map(name=>`<tr><th scope="row">${name}</th>${rates.map(rate=>{const row=turtlebotDropoutRows.find(r=>r.model===name&&Number(r.dropout_fraction)===rate);return `<td>${row?Number(row.macro_f1_mean).toFixed(3):'—'}</td>`}).join('')}</tr>`).join('');
  const svg=document.querySelector('#turtlebot-dropout-chart'), width=720,height=290,left=58,right=24,top=24,bottom=38,plotW=width-left-right,plotH=height-top-bottom;
  const x=v=>left+v/.75*plotW,y=v=>top+(1-v)*plotH,ticks=[0,.25,.5,.75,1];
  const grid=ticks.map(v=>`<line class="axis-line" x1="${left}" x2="${width-right}" y1="${y(v)}" y2="${y(v)}"/><text class="axis-label" x="${left-8}" y="${y(v)+3}" text-anchor="end">${v.toFixed(2)}</text>`).join('');
  const xLabels=rates.map(v=>`<text class="axis-label" x="${x(v)}" y="${height-12}" text-anchor="middle">${Math.round(v*100)}%</text>`).join('');
  const lines=names.map(name=>{const points=rates.map(rate=>{const row=turtlebotDropoutRows.find(r=>r.model===name&&Number(r.dropout_fraction)===rate);return row?{rate,value:Number(row.macro_f1_mean),sd:Number(row.macro_f1_std)}:null}).filter(Boolean);return `<polyline points="${points.map(p=>`${x(p.rate)},${y(p.value)}`).join(' ')}" fill="none" stroke="${colors[name]||'#536779'}" stroke-width="2.5"/>${points.map(p=>`<circle cx="${x(p.rate)}" cy="${y(p.value)}" r="4" fill="${colors[name]||'#536779'}"><title>${name}: ${p.value.toFixed(3)} ± ${p.sd.toFixed(3)}</title></circle>`).join('')}`}).join('');
  svg.innerHTML=`${grid}${lines}${xLabels}<text class="axis-label" x="${left}" y="12">HELD-OUT COMMAND MACRO-F1 · MEAN ± SD</text>`;
}

let trialRows = [];
let trialSummaryRows = [];
const trialMetricInfo = {
  elapsed_s: { title: 'Time to outcome', note: 'Bars show elapsed simulation time. Select a bar to inspect that trial’s path.', label: 'seconds', color: '#23a88f' },
  path_length_px: { title: 'Distance traveled', note: 'Bars show total path length. The timeout run is included; select any bar to inspect its trajectory.', label: 'pixels', color: '#4d8db5' },
  path_efficiency: { title: 'Path directness', note: 'Directness is start-to-final-position displacement divided by distance traveled; 1 is a straight path. Select a bar to inspect it.', label: 'ratio', color: '#d09b45' }
};
let activeTrialId = null;

function renderTrialBars(metric = 'elapsed_s') {
  const info = trialMetricInfo[metric];
  if (!info || !trialSummaryRows.length) return;
  document.querySelector('#trial-chart-title').textContent = info.title;
  document.querySelector('#trial-chart-note').textContent = info.note;
  const svg = document.querySelector('#trial-bar-chart');
  const width = 760, height = 280, left = 48, right = 18, top = 25, bottom = 43;
  const plotW = width - left - right, plotH = height - top - bottom;
  const values = trialSummaryRows.map((row) => Number(row[metric]));
  const maxValue = metric === 'path_efficiency' ? 1 : Math.max(...values) * 1.12;
  const slot = plotW / values.length, barW = Math.min(38, slot * (values.length > 24 ? 0.78 : 0.58));
  const ticks = 4;
  const grid = Array.from({ length: ticks + 1 }, (_, i) => {
    const y = top + plotH - (i / ticks) * plotH;
    return `<line class="trial-chart-grid" x1="${left}" x2="${width - right}" y1="${y}" y2="${y}"/><text class="trial-chart-tick" x="${left - 8}" y="${y + 3}" text-anchor="end">${(maxValue * i / ticks).toFixed(metric === 'path_efficiency' ? 2 : 0)}</text>`;
  }).join('');
  const bars = trialSummaryRows.map((row, index) => {
    const value = Number(row[metric]), x = left + index * slot + (slot - barW) / 2;
    const h = Math.max(2, (value / maxValue) * plotH), y = top + plotH - h;
    const selected = Number(row.trial_id) === Number(activeTrialId);
    const showAnnotation = values.length <= 24 || index % 10 === 0 || index === values.length - 1 || selected;
    const outcomeClass = row.status === 'SUCCESS' ? '' : ' timeout-bar';
    const annotation = showAnnotation ? `<text class="trial-bar-value" x="${x + barW / 2}" y="${Math.max(top + 11, y - 7)}" text-anchor="middle">${value.toFixed(metric === 'path_efficiency' ? 2 : 0)}</text><text class="trial-bar-label" x="${x + barW / 2}" y="${height - 19}" text-anchor="middle">${String(row.trial_id).padStart(2, '0')}</text>` : '';
    return `<g class="trial-bar-group${outcomeClass}${selected ? ' selected' : ''}" data-trial="${row.trial_id}" tabindex="0" role="button" aria-label="Trial ${row.trial_id}: ${value.toFixed(metric === 'path_efficiency' ? 3 : 1)} ${info.label}, ${row.status}"><title>Trial ${row.trial_id}: ${value.toFixed(2)} ${info.label} · ${row.status}. Select to inspect trajectory.</title><rect class="trial-bar" x="${x}" y="${y}" width="${barW}" height="${h}" rx="1" style="--bar-color:${info.color}"/>${annotation}</g>`;
  }).join('');
  svg.innerHTML = `<text class="trial-chart-axis-title" x="${left}" y="13">${info.label.toUpperCase()}</text>${grid}${bars}<text class="trial-chart-axis-title" x="${width - right}" y="${height - 3}" text-anchor="end">TRIAL</text>`;
  svg.querySelectorAll('.trial-bar-group').forEach((bar) => {
    const select = () => {
      document.querySelector('#trial-select').value = bar.dataset.trial;
      drawTrial(bar.dataset.trial);
    };
    bar.addEventListener('click', select);
    bar.addEventListener('keydown', (event) => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } });
  });
}

function renderTrialOutcomes() {
  const success = trialSummaryRows.filter((row) => row.status === 'SUCCESS').length;
  const timeout = trialSummaryRows.length - success;
  const total = Math.max(1, trialSummaryRows.length), circumference = 2 * Math.PI * 55;
  const successArc = circumference * success / total;
  document.querySelector('#outcome-success-count').textContent = success;
  document.querySelector('#outcome-timeout-count').textContent = timeout;
  document.querySelector('#trial-outcome-chart').innerHTML = `<circle class="outcome-track" cx="90" cy="90" r="55"/><circle class="outcome-success-arc" cx="90" cy="90" r="55" stroke-dasharray="${successArc} ${circumference - successArc}" transform="rotate(-90 90 90)"/><text class="outcome-number" x="90" y="87">${success}</text><text class="outcome-caption" x="90" y="106">OF ${total} REACHED</text>`;
}

function drawTrial(trialId) {
  const rows = trialRows.filter((row) => Number(row.trial_id) === Number(trialId));
  if (!rows.length) return;
  const svg = document.querySelector('#sim-chart');
  const left = 38, right = 30, top = 20, bottom = 28, width = 680, height = 360;
  const px = (value) => left + ((Number(value) - 60) / (1040 - 60)) * (width - left - right);
  const py = (value) => top + ((710 - Number(value)) / (710 - 60)) * (height - top - bottom);
  const step = Math.max(1, Math.ceil(rows.length / 360));
  const sampled = rows.filter((_, index) => index % step === 0 || index === rows.length - 1);
  const first = rows[0], last = rows.at(-1);
  const startX = Number(first.start_x_px), startY = Number(first.start_y_px);
  const path = [`${px(startX).toFixed(1)},${py(startY).toFixed(1)}`,
    ...sampled.map((row) => `${px(row.x_px).toFixed(1)},${py(row.y_px).toFixed(1)}`)].join(' ');
  const grid = [0, 1, 2, 3, 4].map((i) => `<line class="sim-grid" x1="${left}" x2="${width - right}" y1="${top + i * ((height - top - bottom) / 4)}" y2="${top + i * ((height - top - bottom) / 4)}"/>`).join('');
  const obstacles = [[280, 180, 25, 260], [500, 330, 250, 25], [830, 150, 25, 250]].map(([x, y, w, h]) => `<rect class="sim-obstacle" x="${px(x)}" y="${py(y + h)}" width="${(w / 980) * (width - left - right)}" height="${(h / 650) * (height - top - bottom)}"/>`).join('');
  svg.innerHTML = `<rect width="${width}" height="${height}" fill="#132235"/>${grid}${obstacles}<polyline class="sim-path" points="${path}"/><circle class="sim-start" cx="${px(startX)}" cy="${py(startY)}" r="7"><title>Start: ${startX}, ${startY}</title></circle><circle class="sim-goal" cx="${px(last.goal_x_px)}" cy="${py(last.goal_y_px)}" r="11"><title>Goal: ${last.goal_x_px}, ${last.goal_y_px}</title></circle><circle class="sim-robot" cx="${px(last.x_px)}" cy="${py(last.y_px)}" r="5"/><text x="${left}" y="${height - 8}" fill="#7f929e" font-size="9" font-family="monospace">SIMULATOR FIELD / PIXELS</text>`;
  document.querySelector('#sim-status').textContent = last.trial_status;
  document.querySelector('#sim-steps').textContent = rows.length.toLocaleString();
  document.querySelector('#sim-path').textContent = `${Number(last.path_length_px).toFixed(1)} px`;
  document.querySelector('#sim-collisions').textContent = last.collision_count;
  activeTrialId = Number(trialId);
  renderTrialBars(document.querySelector('#trial-chart-metric').value);
  document.querySelectorAll('.trial-row').forEach((row) => row.setAttribute('aria-current', String(Number(row.dataset.trial) === Number(trialId))));
}

function renderTrialComparison() {
  const successes = trialSummaryRows.filter((row) => row.status === 'SUCCESS');
  const times = successes.map((row) => Number(row.elapsed_s)).sort((a, b) => a - b);
  const median = times.length ? (times.length % 2 ? times[(times.length - 1) / 2] : (times[times.length / 2 - 1] + times[times.length / 2]) / 2) : null;
  document.querySelector('#trial-count').textContent = trialSummaryRows.length;
  document.querySelector('#trial-success-rate').textContent = trialSummaryRows.length ? `${(successes.length / trialSummaryRows.length * 100).toFixed(0)}%` : '—';
  document.querySelector('#trial-success-count').textContent = `${successes.length} of ${trialSummaryRows.length} goals reached`;
  document.querySelector('#trial-median-time').textContent = median === null ? '—' : `${median.toFixed(1)} s`;
  document.querySelector('#trial-total-collisions').textContent = trialSummaryRows.reduce((sum, row) => sum + Number(row.collision_count), 0);
  renderTrialOutcomes();
  document.querySelector('#trial-table-body').innerHTML = trialSummaryRows.map((row) => `<tr class="trial-row" data-trial="${row.trial_id}" aria-current="false"><td><button type="button" class="trial-row-button">#${row.trial_id}</button></td><td>${row.start_x_px}, ${row.start_y_px}</td><td>${row.goal_x_px}, ${row.goal_y_px}</td><td><span class="trial-outcome ${row.status === 'SUCCESS' ? 'success' : 'timeout'}">${row.status}</span></td><td>${Number(row.steps).toLocaleString()}</td><td>${Number(row.elapsed_s).toFixed(2)}</td><td>${Number(row.path_length_px).toFixed(1)}</td><td>${Number(row.path_efficiency).toFixed(3)}</td><td>${row.collision_count}</td></tr>`).join('');
  document.querySelectorAll('.trial-row').forEach((row) => row.addEventListener('click', () => {
    document.querySelector('#trial-select').value = row.dataset.trial;
    drawTrial(row.dataset.trial);
  }));
}

function modelRankedLabels(model, rawFeatures) {
  const input = rawFeatures.map((value, i) => Number.isFinite(value) ? value : (model.imputer?.[i] ?? 0));
  const x = model.scaler ? input.map((value, i) => (value - model.scaler.mean[i]) / model.scaler.scale[i]) : input;
  const rank = (scores) => model.classes.map((label, i) => ({ label, score: scores[i], i }))
    .sort((a, b) => b.score - a.score || a.i - b.i).map((item) => item.label);
  const walkTree = (tree) => {
    let node = 0;
    while (tree.left[node] !== -1) node = x[tree.feature[node]] <= tree.threshold[node] ? tree.left[node] : tree.right[node];
    return tree.value[node];
  };
  if (model.kind === 'dummy') return [model.prediction];
  if (model.kind === 'logistic') {
    return rank(model.coef.map((row, i) => row.reduce((sum, coefficient, j) => sum + coefficient * x[j], model.intercept[i])));
  }
  if (model.kind === 'random_forest') {
    const probability = Array(model.classes.length).fill(0);
    for (const tree of model.trees) {
      const values = walkTree(tree), total = values.reduce((sum, value) => sum + value, 0) || 1;
      values.forEach((value, i) => { probability[i] += value / total; });
    }
    return rank(probability);
  }
  if (model.kind === 'gradient_boosting') {
    const scores = model.prior.map((prior) => Math.log(Math.max(prior, 1e-15)));
    for (const stage of model.stages) stage.forEach((tree, i) => { scores[i] += model.learning_rate * walkTree(tree)[0]; });
    return rank(scores);
  }
  if (model.kind === 'knn') {
    const nearest = [];
    for (let i = 0; i < model.X.length; i += 1) {
      const sample = model.X[i];
      let distance = 0;
      for (let j = 0; j < x.length; j += 1) distance += (sample[j] - x[j]) ** 2;
      let insertAt = nearest.findIndex((item) => distance < item.distance || (distance === item.distance && i < item.i));
      if (insertAt < 0) insertAt = nearest.length;
      if (insertAt < model.k) {
        nearest.splice(insertAt, 0, { label: model.y[i], distance, i });
        if (nearest.length > model.k) nearest.pop();
      }
    }
    const counts = new Map();
    nearest.forEach((item) => counts.set(item.label, (counts.get(item.label) || 0) + 1));
    return rank(model.classes.map((label) => counts.get(label) || 0));
  }
  if (model.kind === 'svc_rbf') {
    const kernels = model.support_vectors.map((support) => {
      let distance = 0;
      for (let j = 0; j < x.length; j += 1) distance += (support[j] - x[j]) ** 2;
      return Math.exp(-model.gamma * distance);
    });
    const starts = [0];
    model.n_support.forEach((count) => starts.push(starts.at(-1) + count));
    const votes = Array(model.classes.length).fill(0);
    let pair = 0;
    for (let i = 0; i < model.classes.length; i += 1) {
      for (let j = i + 1; j < model.classes.length; j += 1) {
        let decision = model.intercept[pair];
        for (let k = starts[i]; k < starts[i + 1]; k += 1) decision += model.dual_coef[j - 1][k] * kernels[k];
        for (let k = starts[j]; k < starts[j + 1]; k += 1) decision += model.dual_coef[i][k] * kernels[k];
        votes[decision > 0 ? i : j] += 1;
        pair += 1;
      }
    }
    return rank(votes);
  }
  throw new Error(`Unsupported browser model: ${model.kind}`);
}

function initPlanner(bundle) {
  const map = document.querySelector('#planner-map');
  if (!map) return;
  const STEP = 20, radius = 13, rayCount = 24, rayRange = 180;
  const obstacles = [
    { id: 'block-1', x1: 280, y1: 180, x2: 305, y2: 440 },
    { id: 'block-2', x1: 500, y1: 330, x2: 750, y2: 355 },
    { id: 'block-3', x1: 830, y1: 150, x2: 855, y2: 400 }
  ];
  const walls = [
    { x1: 45, y1: 45, x2: 1055, y2: 63 }, { x1: 45, y1: 697, x2: 1055, y2: 715 },
    { x1: 45, y1: 45, x2: 63, y2: 715 }, { x1: 1037, y1: 45, x2: 1055, y2: 715 }
  ];
  const vectors = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
  const palette = ['#4dd6bb', '#f2bd68', '#ff806d', '#71a9ff', '#c095f6'];
  const grid = document.querySelector('#planner-grid');
  for (let x = 80; x <= 1020; x += STEP) grid.innerHTML += `<line x1="${x}" y1="63" x2="${x}" y2="697"/>`;
  for (let y = 80; y <= 680; y += STEP) grid.innerHTML += `<line x1="63" y1="${y}" x2="1037" y2="${y}"/>`;
  const obstacleLayer = document.querySelector('#planner-obstacles');
  const pointsLayer = document.querySelector('#planner-points');
  const pathsLayer = document.querySelector('#planner-model-routes');
  const legend = document.querySelector('#planner-model-legend');
  const status = document.querySelector('#planner-status');
  const startButton = document.querySelector('#planner-animate');
  const stopButton = document.querySelector('#planner-stop');
  const clearButton = document.querySelector('#planner-clear');
  const addButton = document.querySelector('#planner-add-block');
  const removeButton = document.querySelector('#planner-remove-block');
  const placeButton = document.querySelector('#planner-place-points');
  const sizeSelect = document.querySelector('#obstacle-size');
  const speedInput = document.querySelector('#planner-speed');
  const speedOutput = document.querySelector('#planner-speed-label');
  const download = document.querySelector('#planner-download-results');
  const obstacleCount = document.querySelector('#planner-obstacle-count');
  const addedObstacleCount = document.querySelector('#planner-added-count');
  const maxAddedObstacles = 15;
  const models = bundle.models.map((model, i) => ({
    model, color: palette[i % palette.length], state: 'ready', position: null, path: [], history: [], pathLength: 0,
    elapsed: 0, decisions: 0, contacts: 0, blockedSkipped: 0, loopSkipped: 0,
    warnings: 0, wasNear: false, seen: new Set(), recent: [], visits: new Map(),
    bestDistance: Infinity, stalled: 0, blockedStreak: 0
  }));
  legend.innerHTML = models.map((run) => `<span><i style="background:${run.color}"></i>${run.model.name}</span>`).join('');
  let start = null, goal = null, mode = 'points', nextId = 4, frame = null, running = false;

  const collides = (p) => p.x < 63 + radius || p.x > 1037 - radius || p.y < 63 + radius || p.y > 697 - radius ||
    obstacles.some((o) => p.x >= o.x1 - radius && p.x <= o.x2 + radius && p.y >= o.y1 - radius && p.y <= o.y2 + radius);
  const lineCollides = (a, b) => {
    const n = Math.max(2, Math.ceil(Math.hypot(b.x - a.x, b.y - a.y) / 3));
    for (let i = 1; i <= n; i += 1) {
      const t = i / n;
      if (collides({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t })) return true;
    }
    return false;
  };
  const text = (selector, value) => { document.querySelector(selector).textContent = value; };
  function renderObstacles() {
    obstacleLayer.innerHTML = obstacles.map((o) => `<rect x="${o.x1}" y="${o.y1}" width="${o.x2-o.x1}" height="${o.y2-o.y1}" rx="2" class="planner-block${o.userAdded ? ' planner-block-added' : ''}" data-id="${o.id}"/>`).join('');
    obstacleCount.textContent = obstacles.length;
    addedObstacleCount.textContent = `${obstacles.filter((o) => o.userAdded).length} / ${maxAddedObstacles}`;
  }
  function renderPoints() {
    pointsLayer.innerHTML = `${start ? `<circle cx="${start.x}" cy="${start.y}" r="10" class="planner-start-point"/><text x="${start.x+13}" y="${start.y-12}" class="planner-point-label">START</text>` : ''}${goal ? `<circle cx="${goal.x}" cy="${goal.y}" r="10" class="planner-goal-point"/><text x="${goal.x+13}" y="${goal.y-12}" class="planner-point-label">GOAL</text>` : ''}`;
  }
  function renderPaths() {
    pathsLayer.innerHTML = models.map((run) => {
      if (!run.path.length) return '';
      const d = run.path.map((p, i) => `${i ? 'L' : 'M'} ${p.x} ${p.y}`).join(' ');
      const p = run.position;
      return `<path class="planner-route-multi" d="${d}" stroke="${run.color}"/><circle cx="${p.x}" cy="${p.y}" r="5" fill="${run.color}" stroke="#fff" stroke-width="1.5"/>`;
    }).join('');
  }
  function median(values) {
    if (!values.length) return null;
    const a = [...values].sort((x, y) => x - y), mid = Math.floor(a.length / 2);
    return a.length % 2 ? a[mid] : (a[mid - 1] + a[mid]) / 2;
  }
  function renderLineChart(svgId, series, options = {}) {
    const svg = document.querySelector(svgId), width = 720, height = 250;
    const m = { left: 58, right: 18, top: 18, bottom: 38 }, plotW = width - m.left - m.right, plotH = height - m.top - m.bottom;
    const xMax = Math.max(1, ...series.flatMap((s) => s.data.map((p) => p.t)));
    const values = series.flatMap((s) => s.data.map((p) => p.v));
    if (!values.length) {
      svg.innerHTML = `<text class="planner-chart-empty" x="360" y="132" text-anchor="middle">Run all five models to plot ${options.emptyLabel || 'time-series data'}.</text>`;
      return;
    }
    let yMin = options.yMin ?? Math.min(0, ...values), yMax = options.yMax ?? Math.max(1, ...values);
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    if (options.yMin === undefined) { const pad=(yMax-yMin)*.08;yMin-=pad;yMax+=pad; }
    const x = (v) => m.left + (v / xMax) * plotW;
    const y = (v) => m.top + (1 - (v-yMin)/(yMax-yMin)) * plotH;
    const parts = [];
    for (let i=0;i<=4;i+=1) {
      const value=yMin+(yMax-yMin)*i/4, py=y(value);
      parts.push(`<line class="planner-chart-gridline" x1="${m.left}" y1="${py}" x2="${width-m.right}" y2="${py}"/><text class="planner-chart-label" x="${m.left-8}" y="${py+3}" text-anchor="end">${value.toFixed(options.yDecimals ?? 0)}</text>`);
      const time=x(xMax*i/4);
      parts.push(`<text class="planner-chart-label" x="${time}" y="${height-17}" text-anchor="middle">${(xMax*i/4).toFixed(1)}</text>`);
    }
    series.forEach((s) => {
      if (!s.data.length) return;
      const d=s.data.map((p,i)=>`${i?'L':'M'} ${x(p.t).toFixed(1)} ${y(p.v).toFixed(1)}`).join(' ');
      parts.push(`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2" ${s.dash?'stroke-dasharray="5 4"':''}><title>${s.name}</title></path>`);
    });
    parts.push(`<text class="planner-chart-axis" x="${m.left+plotW/2}" y="${height-2}" text-anchor="middle">SIMULATED TIME · SECONDS</text>`);
    parts.push(`<text class="planner-chart-axis" transform="translate(13 ${m.top+plotH/2}) rotate(-90)" text-anchor="middle">${options.axisLabel || 'VALUE'}</text>`);
    svg.innerHTML=parts.join('');
  }
  function renderValidationChart() {
    const svg=document.querySelector('#planner-validation-chart'), width=720, height=250, left=190, right=45, barH=24, gap=17;
    const plotW=width-left-right, rows=models.length, top=(height-(rows*(barH+gap)-gap))/2;
    const parts=[];
    for(let tick=0;tick<=1.0001;tick+=.2){const px=left+tick*plotW;parts.push(`<line class="planner-chart-gridline" x1="${px}" y1="${top-8}" x2="${px}" y2="${height-24}"/><text class="planner-chart-label" x="${px}" y="${height-7}" text-anchor="middle">${tick.toFixed(1)}</text>`);}
    models.forEach((run,i)=>{
      const yy=top+i*(barH+gap), score=Number(run.model.validation_macro_f1)||0;
      parts.push(`<text class="planner-chart-label" x="${left-10}" y="${yy+16}" text-anchor="end">${run.model.name}</text><rect x="${left}" y="${yy}" width="${Math.max(0,score)*plotW}" height="${barH}" rx="3" fill="${run.color}"/><text class="planner-chart-value" x="${left+score*plotW+7}" y="${yy+16}">${score.toFixed(3)}</text>`);
    });
    svg.innerHTML=parts.join('');
  }
  function renderDiagnostics() {
    const positions=[], speeds=[];
    for(const run of models){
      positions.push({name:`${run.model.name} · X`,color:run.color,data:run.history.map((p)=>({t:p.t,v:p.x}))});
      positions.push({name:`${run.model.name} · Y`,color:run.color,dash:true,data:run.history.map((p)=>({t:p.t,v:p.y}))});
      speeds.push({name:run.model.name,color:run.color,data:run.history.map((p)=>({t:p.t,v:p.speed}))});
    }
    renderLineChart('#planner-position-chart',positions,{yMin:0,yMax:1100,axisLabel:'MAP POSITION · PX',emptyLabel:'position traces'});
    renderLineChart('#planner-speed-chart',speeds,{axisLabel:'GOALWARD SPEED · PX/S',yDecimals:0,emptyLabel:'speed traces'});
  }
  function csvCell(value) { return `"${String(value).replaceAll('"', '""')}"`; }
  function renderResults() {
    const successful = models.filter((r) => r.state === 'success');
    const completed = models.filter((r) => ['success', 'failed', 'stopped'].includes(r.state));
    text('#planner-successes', `${successful.length} / ${models.length}`);
    text('#planner-success-rate', `${(100 * successful.length / models.length).toFixed(0)}%`);
    const medianPath = median(successful.map((r) => r.pathLength));
    text('#planner-median-path', medianPath === null ? '—' : `${medianPath.toFixed(0)} px`);
    const fastest = [...successful].sort((a, b) => a.elapsed - b.elapsed)[0];
    text('#planner-fastest', fastest ? `${fastest.model.name} · ${fastest.elapsed.toFixed(1)} s` : '—');
    text('#planner-filtered', models.reduce((n, r) => n + r.blockedSkipped, 0));
    text('#planner-loop-skipped', models.reduce((n, r) => n + r.loopSkipped, 0));
    const rows = models.map((r) => `<article class="planner-result-row"><header><i style="background:${r.color}"></i><b>${r.model.name}</b><span class="planner-outcome planner-outcome-${r.state}">${r.state === 'success' ? 'REACHED' : r.state === 'failed' ? r.reason : r.state === 'stopped' ? 'STOPPED' : r.state === 'running' ? 'RUNNING' : 'READY'}</span></header><div><span>${r.decisions} moves</span><span>${r.pathLength.toFixed(0)} px</span><span>${r.elapsed.toFixed(1)} s</span><span>${r.contacts} contacts</span><span>${r.blockedSkipped} blocked · ${r.loopSkipped} loop skips</span></div></article>`).join('');
    document.querySelector('#planner-model-results').innerHTML = rows;
    if (!completed.length) text('#planner-analysis', 'Set start and goal points to compare policies under identical conditions.');
    else {
      const mean = successful.length ? successful.reduce((sum, r) => sum + r.pathLength, 0) / successful.length : null;
      const sd = successful.length > 1 ? Math.sqrt(successful.reduce((sum, r) => sum + (r.pathLength - mean) ** 2, 0) / (successful.length - 1)) : 0;
      const shortest = [...successful].sort((a, b) => a.pathLength - b.pathLength)[0];
      text('#planner-analysis', successful.length
        ? `${successful.length} of ${models.length} reached the goal. Successful path length: mean ${mean.toFixed(0)} px, median ${medianPath.toFixed(0)} px, sample SD ${sd.toFixed(0)} px. Shortest: ${shortest.model.name} (${shortest.pathLength.toFixed(0)} px). Descriptive statistics for this single map; no significance test.`
        : `No model reached the goal in this run. ${models.length} policies were compared on the same map.`);
    }
    const header = ['model','outcome','moves','simulated_seconds','path_length_px','contacts','blocked_choices_skipped','recent_loop_choices_skipped','sensor_warnings','obstacles_sensed','trajectory_time_x_y_goalward_speed'];
    if (!running) {
      const csv = [header, ...models.map((r) => [r.model.name, r.state === 'success' ? 'SUCCESS' : (r.reason || r.state).toUpperCase(), r.decisions, r.elapsed.toFixed(2), r.pathLength.toFixed(1), r.contacts, r.blockedSkipped, r.loopSkipped, r.warnings, r.seen.size, r.history.map((p) => `${p.t.toFixed(2)}:${p.x}:${p.y}:${p.speed.toFixed(2)}`).join('|')])].map((row) => row.map(csvCell).join(',')).join('\n');
      if (download.dataset.url) URL.revokeObjectURL(download.dataset.url);
      const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
      download.href = url; download.dataset.url = url; download.hidden = completed.length === 0;
    }
  }
  function setMode(next) {
    mode = next;
    clearRun();
    addButton.setAttribute('aria-pressed', String(mode === 'add'));
    removeButton.setAttribute('aria-pressed', String(mode === 'remove'));
    placeButton.setAttribute('aria-pressed', String(mode === 'points'));
    if (mode === 'add') status.textContent = 'Add-block mode: click open floor to place an obstacle.';
    if (mode === 'remove') status.textContent = 'Remove-block mode: click a block to remove it.';
  }
  function clearRun() {
    if (frame !== null) cancelAnimationFrame(frame);
    frame = null; running = false; start = null; goal = null;
    models.forEach((r) => Object.assign(r, { state: 'ready', position: null, path: [], history: [], pathLength: 0, elapsed: 0, decisions: 0, contacts: 0, blockedSkipped: 0, loopSkipped: 0, warnings: 0, wasNear: false, seen: new Set(), recent: [], visits: new Map(), bestDistance: Infinity, stalled: 0, blockedStreak: 0 }));
    pathsLayer.innerHTML = ''; pointsLayer.innerHTML = ''; renderResults(); renderDiagnostics();
    startButton.disabled = true; stopButton.disabled = true; startButton.textContent = `▶ Run all ${models.length} models`;
    status.textContent = 'Click two open points for a shared start and goal. All five models will run together.';
  }
  function addObstacle(point) {
    if (obstacles.filter((o) => o.userAdded).length >= maxAddedObstacles) {
      status.textContent = `The map allows at most ${maxAddedObstacles} user-added blocks. Remove a block before adding another.`; return;
    }
    const [w, h] = sizeSelect.value.split('x').map(Number), x = Math.round(point.x / 10) * 10, y = Math.round(point.y / 10) * 10;
    const block = { id: `block-${nextId++}`, x1:x-w/2, y1:y-h/2, x2:x+w/2, y2:y+h/2, userAdded:true };
    if (block.x1 < 65 || block.y1 < 65 || block.x2 > 1035 || block.y2 > 695 || obstacles.some((o) => block.x1 < o.x2+3 && block.x2 > o.x1-3 && block.y1 < o.y2+3 && block.y2 > o.y1-3)) {
      status.textContent = 'That block overlaps an obstacle or arena border. Choose another open location.'; return;
    }
    obstacles.push(block); renderObstacles(); clearRun();
  }
  function removeObstacle(point) {
    const i = obstacles.findIndex((o) => point.x >= o.x1 && point.x <= o.x2 && point.y >= o.y1 && point.y <= o.y2);
    if (i < 0) { status.textContent = 'Click inside an obstacle block to remove it.'; return; }
    obstacles.splice(i, 1); renderObstacles(); clearRun();
  }
  function featuresFor(run) {
    const at = run.position, values = [], near = [];
    for (let i = 0; i < rayCount; i += 1) {
      const angle = i * Math.PI * 2 / rayCount, dx = Math.cos(angle), dy = Math.sin(angle);
      let distance = rayRange;
      for (const rect of [...obstacles, ...walls]) {
        let lo = 0, hi = distance, valid = true;
        for (const [origin, direction, minimum, maximum] of [[at.x,dx,rect.x1,rect.x2],[at.y,dy,rect.y1,rect.y2]]) {
          if (Math.abs(direction) < 1e-12) { if (origin < minimum || origin > maximum) { valid = false; break; } }
          else { const a=(minimum-origin)/direction,b=(maximum-origin)/direction;lo=Math.max(lo,Math.min(a,b));hi=Math.min(hi,Math.max(a,b));if(hi<lo){valid=false;break;} }
        }
        if (valid && hi >= 0 && lo <= distance) { distance=Math.max(0,lo);if(obstacles.includes(rect))run.seen.add(rect.id); }
      }
      values.push(distance / rayRange); near.push(distance <= 45);
    }
    const close = near.some(Boolean);
    if (close && !run.wasNear) run.warnings += 1;
    run.wasNear = close;
    return [(goal.x-at.x)/1000,(goal.y-at.y)/700,at.x/1100,at.y/760,...values];
  }
  function stepRun(run) {
    const ranked = modelRankedLabels(run.model, featuresFor(run));
    let selected = null, fallback = null;
    const recent = new Set(run.recent);
    for (const label of ranked) {
      const vector = vectors[Number(label)]; if (!vector) continue;
      const next = { x:run.position.x+vector[0]*STEP, y:run.position.y+vector[1]*STEP };
      if (lineCollides(run.position,next)) { run.blockedSkipped += 1; continue; }
      const key = `${next.x},${next.y}`;
      if (recent.has(key)) {
        run.loopSkipped += 1;
        if (!fallback || (run.visits.get(key)||0) < (run.visits.get(`${fallback.next.x},${fallback.next.y}`)||0)) fallback={label,vector,next};
        continue;
      }
      selected={label,vector,next}; break;
    }
    selected ||= fallback;
    run.decisions += 1;
    if (!selected) {
      run.contacts += 1; run.blockedStreak += 1; run.elapsed += STEP/95;
      run.history.push({x:run.position.x,y:run.position.y,t:run.elapsed,speed:0});
      if (run.blockedStreak >= 8) { run.state='failed'; run.reason='BLOCKED'; }
      return;
    }
    run.blockedStreak=0;
    const distance=Math.hypot(selected.next.x-run.position.x,selected.next.y-run.position.y);
    const oldDistance=Math.hypot(goal.x-run.position.x,goal.y-run.position.y), deltaTime=distance/95;
    run.elapsed += deltaTime; run.pathLength += distance; run.position=selected.next; run.path.push(run.position);
    const key=`${run.position.x},${run.position.y}`; run.visits.set(key,(run.visits.get(key)||0)+1);
    run.recent.push(key); if(run.recent.length>48)run.recent.shift();
    const remaining=Math.hypot(goal.x-run.position.x,goal.y-run.position.y);
    run.history.push({x:run.position.x,y:run.position.y,t:run.elapsed,speed:(oldDistance-remaining)/deltaTime});
    if(remaining<=20){run.state='success';return;}
    if(remaining<run.bestDistance-10){run.bestDistance=remaining;run.stalled=0;}else run.stalled+=1;
    if(run.stalled>=120){run.state='failed';run.reason='STALLED';}
    else if(run.elapsed>=60){run.state='failed';run.reason='TIME LIMIT';}
  }
  function stopAll(message) {
    if(frame!==null)cancelAnimationFrame(frame); frame=null; running=false;
    models.forEach((r)=>{if(r.state==='running'){r.state='stopped';r.reason='STOPPED';}});
    stopButton.disabled=true;startButton.disabled=!(start&&goal);startButton.textContent=`▶ Run all ${models.length} models`;
    renderPaths();renderResults();if(message)status.textContent=message;
  }
  function runAll() {
    if(!start||!goal||running)return;
    models.forEach((r)=>{Object.assign(r,{state:'running',position:{...start},path:[{...start}],history:[{x:start.x,y:start.y,t:0,speed:0}],pathLength:0,elapsed:0,decisions:0,contacts:0,blockedSkipped:0,loopSkipped:0,warnings:0,wasNear:false,seen:new Set(),recent:[`${start.x},${start.y}`],visits:new Map([[`${start.x},${start.y}`,1]]),bestDistance:Math.hypot(goal.x-start.x,goal.y-start.y),stalled:0,blockedStreak:0});});
    running=true;startButton.disabled=true;stopButton.disabled=false;startButton.textContent='● Running all models';
    status.textContent=`Running ${models.length} policies together from the same start and goal.`;renderResults();renderPaths();renderDiagnostics();
    let previous=null,accumulator=0,lastSummary=0;const interval=STEP/95;
    const animate=(now)=>{
      if(previous===null)previous=now;
      accumulator+=Math.min((now-previous)/1000,.1)*Number(speedInput.value);previous=now;
      let steps=0;
      while(accumulator>=interval&&steps<8){
        accumulator-=interval;steps+=1;
        for(const run of models)if(run.state==='running')stepRun(run);
        if(models.every((r)=>r.state!=='running')){stopAll(`Comparison complete: ${models.filter((r)=>r.state==='success').length} of ${models.length} models reached the goal.`);return;}
      }
      renderPaths();renderDiagnostics();
      if(now-lastSummary>350){renderResults();lastSummary=now;}
      frame=requestAnimationFrame(animate);
    };
    frame=requestAnimationFrame(animate);
  }
  renderObstacles();renderResults();renderValidationChart();renderDiagnostics();
  map.addEventListener('click',(event)=>{
    if(running)return;
    const p=map.createSVGPoint();p.x=event.clientX;p.y=event.clientY;
    const q=p.matrixTransform(map.getScreenCTM().inverse());
    if(mode==='add'){addObstacle(q);return;}
    if(mode==='remove'){removeObstacle(q);return;}
    const point={x:80+Math.round((q.x-80)/STEP)*STEP,y:80+Math.round((q.y-80)/STEP)*STEP};
    if(collides(point)){status.textContent='That point is too close to a wall or obstacle. Choose open floor.';return;}
    if(!start||goal){start=point;goal=null;models.forEach((r)=>Object.assign(r,{state:'ready',reason:'',position:null,path:[],history:[],pathLength:0,elapsed:0,decisions:0,contacts:0,blockedSkipped:0,loopSkipped:0,warnings:0,wasNear:false,seen:new Set(),recent:[],visits:new Map(),bestDistance:Infinity,stalled:0,blockedStreak:0}));pathsLayer.innerHTML='';startButton.disabled=true;status.textContent='Start set. Click another open point to set the shared goal.';}
    else{goal=point;startButton.disabled=false;status.textContent=`Shared goal set. Run all ${models.length} models to compare their paths.`;}
    renderPoints();renderResults();renderDiagnostics();
  });
  clearButton.addEventListener('click',clearRun);
  addButton.addEventListener('click',()=>setMode(mode==='add'?'points':'add'));
  removeButton.addEventListener('click',()=>setMode(mode==='remove'?'points':'remove'));
  placeButton.addEventListener('click',()=>setMode('points'));
  startButton.addEventListener('click',runAll);stopButton.addEventListener('click',()=>stopAll('Comparison stopped. Run all models to restart.'));
  speedInput.addEventListener('input',()=>{speedOutput.textContent=`${speedInput.value}×`;});
  startButton.textContent=`▶ Run all ${models.length} models`;
}

async function init() {
  renderClassBars();
  initDepthInteractions();
  loadRealRunValidation();
  loadPhysicalRunQuality();
  document.querySelector('#trial-chart-note').textContent='Loading saved trial records…';
  document.querySelector('#sim-chart').innerHTML='<text class="planner-chart-empty" x="340" y="180" text-anchor="middle">Loading the recorded trajectories…</text>';
  document.querySelector('#trial-bar-chart').innerHTML='<text class="planner-chart-empty" x="380" y="145" text-anchor="middle">Loading the 100 trial results…</text>';
  document.querySelector('#trial-outcome-chart').innerHTML='<text class="planner-chart-empty" x="90" y="94" text-anchor="middle">Loading</text>';
  document.querySelector('#trial-table-body').innerHTML='<tr><td colspan="8">Loading the 100 saved trial outcomes…</td></tr>';
  try {
    const [models, robustness, ordered, barn, turtlebotDropout, steps, summaries] = await Promise.all([
      csv(DATA.models), csv(DATA.robustness), csv(DATA.orderedHoldout), csv(DATA.barnEvaluation), csv(DATA.turtlebotDropout), csv(DATA.trials), csv(DATA.trialSummary)
    ]);
    modelRows = models; robustnessRows = robustness; orderedHoldoutRows = ordered; barnEvaluationRows = barn; turtlebotDropoutRows = turtlebotDropout; trialRows = steps; trialSummaryRows = summaries;
    renderModels();
    document.querySelector('#metric-select').addEventListener('change', (event) => renderModels(event.target.value));
    const discussionSelect = document.querySelector('#discussion-model');
    discussionSelect.innerHTML = modelRows.map((row) => `<option value="${row.model}">${row.model}</option>`).join('');
    discussionSelect.value = modelRows[0]?.model || '';
    discussionSelect.addEventListener('change', (event) => showModelDetail(event.target.value, document.querySelector('#metric-select').value));
    renderRobustnessChart();
    document.querySelector('#robustness-mode').addEventListener('change', event => renderRobustnessChart(event.target.value));
    renderOrderedHoldout();
    renderBarnEvaluation();
    renderTurtlebotDropout();
    const trialSelect = document.querySelector('#trial-select');
    const ids = trialSummaryRows.map((row) => row.trial_id).filter(Boolean).sort((a, b) => Number(a) - Number(b));
    trialSelect.innerHTML = ids.map((id) => `<option value="${id}">#${id}</option>`).join('');
    trialSelect.addEventListener('change', () => drawTrial(trialSelect.value));
    renderTrialComparison();
    document.querySelector('#trial-chart-metric').addEventListener('change', (event) => renderTrialBars(event.target.value));
    if (ids.length) { trialSelect.value = ids[0]; drawTrial(ids[0]); }
  } catch (error) {
    document.querySelector('#model-table').innerHTML = `<p class="loading">${error.message}. Serve the website folder over HTTP; see website/README.md.</p>`;
    document.querySelector('#noise-score').textContent = 'unavailable';
    document.querySelector('#sim-status').textContent = 'unavailable';
    document.querySelector('#trial-chart-note').textContent = `Trial results could not load: ${error.message}`;
  }

  // The fitted browser policies are a large bundle. Load them only as the
  // planner approaches the viewport, keeping the results and trial explorer
  // responsive while the simulation model is downloaded and parsed.
  const plannerSection=document.querySelector('#path-planner');
  let plannerLoaded=false;
  const loadPlanner=()=>{
    if(plannerLoaded)return;
    plannerLoaded=true;
    document.querySelector('#planner-status').textContent='Loading fitted navigation policies…';
    fetch(DATA.browserModels)
      .then(response => { if (!response.ok) throw new Error(`HTTP ${response.status}`); return response.json(); })
      .then(initPlanner)
      .catch(error => { document.querySelector('#planner-status').textContent = `ML simulator unavailable: ${error.message}`; });
  };
  if('IntersectionObserver' in window){
    const observer=new IntersectionObserver(entries=>{if(entries.some(entry=>entry.isIntersecting)){observer.disconnect();loadPlanner();}},{rootMargin:'600px'});
    observer.observe(plannerSection);
  } else loadPlanner();
}

init();

function initDepthInteractions() {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches || !window.matchMedia('(hover: hover)').matches) return;
  document.querySelectorAll('.hero-visual, .method-card, .class-panel, .field-figure, .result-card, .ordered-holdout-card, .trial-chart-card, .trial-table-wrap, .readiness-card, .planner-chart-card, .planner-map-panel, .planner-controls, .source-list, .comparison-shell, .interpretation').forEach((card) => {
    card.addEventListener('pointermove', (event) => {
      const bounds = card.getBoundingClientRect();
      const x = (event.clientX - bounds.left) / bounds.width - .5;
      const y = (event.clientY - bounds.top) / bounds.height - .5;
      card.style.setProperty('--tilt-x', `${(-y * 3).toFixed(2)}deg`);
      card.style.setProperty('--tilt-y', `${(x * 4).toFixed(2)}deg`);
    });
    card.addEventListener('pointerleave', () => {
      card.style.removeProperty('--tilt-x');
      card.style.removeProperty('--tilt-y');
    });
  });
}
