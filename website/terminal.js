(() => {
  const panel = document.querySelector('#research-terminal');
  const toggle = document.querySelector('#terminal-toggle');
  const close = document.querySelector('.terminal-close');
  const form = document.querySelector('#terminal-form');
  const input = document.querySelector('#terminal-input');
  const output = document.querySelector('#terminal-output');
  if (!panel || !toggle || !form || !input || !output) return;

  const responses = {
    help: 'Commands: help, about, projects, skills, models, results, limits, clear. Use these to inspect this research site.',
    about: 'This project compares classical machine-learning classifiers for UCI wall-following robot sensor snapshots. Classification performance does not by itself establish safe navigation.',
    models: 'The evaluated classifiers are Dummy baseline, Logistic Regression, RBF SVM, K-Nearest Neighbors, Random Forest, and Gradient Boosting. See Methods for the model setup and held-out metrics.',
    results: 'The Results section contains the model comparison, noise robustness curve, confusion matrix, ordered split, and recorded simulation summaries. Read each chart with its evaluation caveats.',
    limits: 'Key limits: UCI records do not contain independent run IDs; noise/dropout are software perturbations; the browser planner uses synthetic sensor readings; no physical safety claim is made.',
    projects: 'Explore the Results section for model evidence, the ML simulation for goal-conditioned policy demonstrations, and Reproduce & Trace for code and data sources.',
    skills: 'Methods shown here include sensor-data preprocessing, stratified model evaluation, macro-F1 comparison, confusion-matrix analysis, noise/dropout robustness checks, and goal-conditioned policy simulation.'
  };
  const append = (text, className = 'terminal-line') => {
    const line = document.createElement('p');
    line.className = className;
    line.textContent = text;
    output.append(line);
    output.scrollTop = output.scrollHeight;
  };
  const setOpen = open => {
    panel.hidden = !open;
    toggle.setAttribute('aria-expanded', String(open));
    if (open) input.focus();
  };
  toggle.addEventListener('click', () => setOpen(panel.hidden));
  close?.addEventListener('click', () => { setOpen(false); toggle.focus(); });
  addEventListener('keydown', event => {
    if (event.key === '/' && panel.hidden && !/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || '')) {
      event.preventDefault();
      setOpen(true);
    }
    if (event.key === 'Escape' && !panel.hidden) { setOpen(false); toggle.focus(); }
  });
  form.addEventListener('submit', event => {
    event.preventDefault();
    const command = input.value.trim().toLowerCase();
    if (!command) return;
    append(`visitor@robot-lab:~$ ${command}`, 'terminal-command');
    if (command === 'clear') output.replaceChildren();
    else append(responses[command] || `Command not found: ${command}. Type help to see available commands.`, responses[command] ? 'terminal-response' : 'terminal-error');
    input.value = '';
    input.focus();
  });
})();
