/* RL Prompt Optimizer — chat front end */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const STORE_KEY = 'rlpo.chats.v1';
  const PREF_KEY = 'rlpo.prefs.v1';

  const state = {
    providers: {},              // name -> {models, has_key}
    provider: 'Gemini',
    model: '',
    chats: load(STORE_KEY, []),
    currentId: null,
    busy: false,
    review: false,
  };
  const prefs = load(PREF_KEY, {});

  // ---------- utils ----------
  function load(k, d) { try { return JSON.parse(localStorage.getItem(k)) ?? d; } catch { return d; } }
  function save() { localStorage.setItem(STORE_KEY, JSON.stringify(state.chats)); }
  function savePrefs() { localStorage.setItem(PREF_KEY, JSON.stringify({ provider: state.provider, model: state.model, review: state.review })); }
  const uid = () => Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
  const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
  const icon = d => `<svg viewBox="0 0 24 24"><path d="${d}"/></svg>`;
  const ICONS = {
    spark: 'M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z',
    chart: 'M4 20V10M10 20V4M16 20v-7M22 20H2',
    chat: 'M21 12a8 8 0 01-11.6 7.1L4 20l1-4.6A8 8 0 1121 12z',
    trash: 'M4 7h16M10 11v6M14 11v6M6 7l1 13h10l1-13M9 7V4h6v3',
    chev: 'M9 6l6 6-6 6',
    warn: 'M12 9v4M12 17h.01M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z',
  };

  function toast(msg) {
    const t = $('#toast'); t.textContent = msg; t.classList.add('show');
    clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.remove('show'), 3200);
  }

  async function api(path, body) {
    const res = await fetch(path, body ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) } : {});
    let data = null; try { data = await res.json(); } catch { }
    if (!res.ok) throw new Error((data && data.detail) || `HTTP ${res.status}`);
    return data;
  }

  // ---------- minimal, safe markdown ----------
  function md(src) {
    const blocks = [];
    let text = String(src ?? '').replace(/\r\n/g, '\n');
    text = text.replace(/```([\w+-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      blocks.push(`<pre><code>${esc(code.replace(/\n$/, ''))}</code></pre>`);
      return `\u0000${blocks.length - 1}\u0000`;
    });
    const inline = s => esc(s)
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
      .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, '$1<em>$2</em>')
      .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    const lines = text.split('\n');
    const out = []; let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (/^\u0000\d+\u0000$/.test(line.trim())) { out.push(blocks[+line.trim().slice(1, -1)]); i++; continue; }
      if (!line.trim()) { i++; continue; }
      let m;
      if ((m = line.match(/^(#{1,4})\s+(.*)/))) { out.push(`<h${m[1].length}>${inline(m[2])}</h${m[1].length}>`); i++; continue; }
      if (/^(-{3,}|\*{3,})$/.test(line.trim())) { out.push('<hr>'); i++; continue; }
      if (/^>\s?/.test(line)) {
        const q = []; while (i < lines.length && /^>\s?/.test(lines[i])) q.push(lines[i++].replace(/^>\s?/, ''));
        out.push(`<blockquote>${inline(q.join(' '))}</blockquote>`); continue;
      }
      if (/^\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\|?\s*:?-{2,}/.test(lines[i + 1])) {
        const row = l => l.trim().replace(/^\||\|$/g, '').split('|').map(c => c.trim());
        const head = row(line); i += 2; const body = [];
        while (i < lines.length && /^\|.*\|\s*$/.test(lines[i])) body.push(row(lines[i++]));
        out.push(`<table><thead><tr>${head.map(h => `<th>${inline(h)}</th>`).join('')}</tr></thead><tbody>${body.map(r => `<tr>${r.map(c => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`);
        continue;
      }
      if (/^\s*([-*+]|\d+[.)])\s+/.test(line)) {
        const ordered = /^\s*\d+[.)]/.test(line); const items = [];
        while (i < lines.length && /^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) {
          let item = lines[i++].replace(/^\s*([-*+]|\d+[.)])\s+/, '');
          while (i < lines.length && /^\s{2,}\S/.test(lines[i]) && !/^\s*([-*+]|\d+[.)])\s+/.test(lines[i])) item += ' ' + lines[i++].trim();
          items.push(`<li>${inline(item)}</li>`);
        }
        out.push(ordered ? `<ol>${items.join('')}</ol>` : `<ul>${items.join('')}</ul>`); continue;
      }
      const para = [];
      while (i < lines.length && lines[i].trim() && !/^(#{1,4}\s|>|\s*([-*+]|\d+[.)])\s+|\u0000)/.test(lines[i])) para.push(lines[i++]);
      if (para.length) out.push(`<p>${inline(para.join('\n')).replace(/\n/g, '<br>')}</p>`); else i++;
    }
    return out.join('');
  }

  // ---------- models / provider ----------
  async function initModels() {
    try {
      const { providers } = await api('/api/models');
      providers.forEach(p => state.providers[p.name] = p);
      ['Gemini', 'Groq', 'Nvidia'].forEach(n => {
        const btn = $(`[data-provider="${n}"]`);
        if (state.providers[n] && !state.providers[n].has_key) { btn.classList.add('nokey'); btn.title = `${n} key missing in .env`; }
      });
    } catch (e) { toast('Could not load models: ' + e.message); }
    setProvider(prefs.provider || 'Gemini', prefs.model);
  }

  function setProvider(name, model) {
    state.provider = name;
    document.querySelectorAll('.provider-btn').forEach(b => b.classList.toggle('active', b.dataset.provider === name));
    moveGlider();
    const sel = $('#model-select');
    const models = state.providers[name]?.models || [];
    sel.innerHTML = models.map(m => `<option value="${esc(m)}">${esc(m)}</option>`).join('');
    state.model = models.includes(model) ? model : models[0] || '';
    sel.value = state.model;
    savePrefs();
  }

  function moveGlider() {
    const active = $('.provider-btn.active'); const g = $('#provider-glider');
    if (!active) return;
    g.style.width = active.offsetWidth + 'px';
    g.style.transform = `translateX(${active.offsetLeft - 4}px)`;
  }

  // ---------- policy panel ----------
  function renderPolicy(stats, flash) {
    if (!stats) return;
    const rows = Object.entries(stats).sort((a, b) => b[1].prob - a[1].prob);
    const maxP = Math.max(...rows.map(([, v]) => v.prob), 0.0001);
    $('#policy-list').innerHTML = rows.map(([k, v]) => `
      <div class="policy-row ${k === flash ? 'flash' : ''}" title="Q = ${v.q}, tried ${v.n}×, selection probability ${(v.prob * 100).toFixed(1)}%">
        <div class="top"><b>${esc(v.label)}</b><span class="meta">${(v.prob * 100).toFixed(0)}% · Q ${v.q >= 0 ? '+' : ''}${v.q.toFixed(2)} · n${v.n}</span></div>
        <div class="bar"><i style="width:${(v.prob / maxP * 100).toFixed(1)}%"></i></div>
      </div>`).join('');
  }
  async function loadPolicy() { try { renderPolicy(await api('/api/policy')); } catch { } }

  // ---------- chats ----------
  const current = () => state.chats.find(c => c.id === state.currentId);

  function renderChatList() {
    const list = $('#chat-list');
    if (!state.chats.length) { list.innerHTML = '<div class="empty-list">No chats yet</div>'; return; }
    list.innerHTML = state.chats.map(c => `
      <button class="chat-item ${c.id === state.currentId ? 'active' : ''}" data-id="${c.id}">
        <span>${esc(c.title)}</span>
        <i class="del" data-del="${c.id}" title="Delete">${icon(ICONS.trash)}</i>
      </button>`).join('');
  }

  function openChat(id) {
    state.currentId = id;
    renderChatList(); renderMessages();
  }

  function newChat() {
    if (state.busy) return;
    state.currentId = null;
    renderChatList(); renderMessages();
    $('#prompt-input').focus();
  }

  function renderMessages() {
    const chat = current();
    const box = $('#messages');
    box.innerHTML = '';
    $('#welcome').style.display = chat && chat.turns.length ? 'none' : '';
    if (!chat) return;
    chat.turns.forEach(t => box.appendChild(turnEl(t)));
    scrollDown(false);
  }

  function scrollDown(smooth = true) {
    const s = $('#chat-scroll');
    requestAnimationFrame(() => s.scrollTo({ top: s.scrollHeight, behavior: smooth ? 'smooth' : 'auto' }));
  }

  function refreshTurn(turn) {
    const old = document.getElementById('turn-' + turn.id);
    const el = turnEl(turn);
    if (old) old.replaceWith(el); else $('#messages').appendChild(el);
    save();
  }

  // ---------- turn rendering ----------
  function turnEl(t) {
    const wrap = document.createElement('div');
    wrap.id = 'turn-' + t.id;
    wrap.style.display = 'contents';
    wrap.innerHTML = `
      <div class="msg-user"><div class="bubble">${esc(t.prompt)}</div></div>
      <div class="msg-ai">
        <div class="ai-avatar ${['optimizing', 'executing'].includes(t.status) ? 'spin' : ''}">RL</div>
        <div class="ai-body"></div>
      </div>`;
    const body = $('.ai-body', wrap);
    body.appendChild(optCard(t));
    if (t.status === 'error') body.appendChild(errCard(t.error));
    if (t.status === 'executing') body.appendChild(execPending(t));
    if (t.exec) { body.appendChild(responseCard(t)); body.appendChild(effCard(t)); }
    return wrap;
  }

  function optCard(t) {
    const card = document.createElement('div');
    card.className = 'card';
    const o = t.opt;
    const meta = `${esc(t.provider)} · ${esc(t.model)}${o ? ` · ${o.seconds}s` : ''}`;
    if (!o) {
      card.innerHTML = `
        <div class="card-head"><div class="title">${icon(ICONS.spark)}RL prompt optimisation</div><div class="meta">${meta}</div></div>
        <div class="steps">
          ${step('Sampling rewrite strategies from the policy', t.status === 'optimizing' ? 'active' : 'fail')}
          ${step('Generating candidate prompts', '')}
          ${step('Scoring candidates (reward)', '')}
        </div>`;
      return card;
    }
    const up = o.optimized_score != null && o.original_score != null && o.optimized_score > o.original_score;
    const editable = t.status === 'review';
    card.innerHTML = `
      <div class="card-head"><div class="title">${icon(ICONS.spark)}RL prompt optimisation</div><div class="meta">${meta}</div></div>
      <div class="compare">
        <div class="orig">
          <div class="lbl">Prompt given <span class="score-pill">${fmtScore(o.original_score)}</span></div>
          <div class="prompt-text">${esc(o.original)}</div>
        </div>
        <div class="opt">
          <div class="lbl">RL-optimised prompt <span class="score-pill ${up ? 'up' : ''}">${fmtScore(o.optimized_score)}</span></div>
          <div class="prompt-text" id="opt-text-${t.id}">${esc(t.finalPrompt ?? o.optimized)}</div>
        </div>
      </div>
      <div class="strategy-row">
        Selected action: <span class="badge">${esc(o.strategy_label)}</span>
        ${o.unchanged ? '<span>Your original prompt already scored best, so it is sent unchanged.</span>' : ''}
        ${t.sentOriginal ? '<span>You chose to send the original prompt only.</span>' : ''}
        ${t.edited ? '<span>Edited by you before sending.</span>' : ''}
        ${o.reason ? `<div class="reason">${esc(o.reason)}</div>` : ''}
      </div>
      <details class="cands">
        <summary>${icon(ICONS.chev)}${o.candidates.length} candidates scored (policy sampled: ${o.sampled_strategies.map(esc).join(', ')})</summary>
        ${o.candidates.map(c => `
          <div class="cand ${c.prompt === o.optimized ? 'win' : ''}">
            <div class="ch"><b>${esc(c.label)}${c.prompt === o.optimized ? ' ✓ selected' : ''}</b><span>${fmtScore(c.score)}</span></div>
            <p>${esc(c.prompt)}</p>
          </div>`).join('')}
      </details>
      ${editable ? `
        <div class="actions">
          <button class="btn primary" data-act="send-opt">Send optimised</button>
          <button class="btn" data-act="edit">Edit optimised</button>
          <button class="btn" data-act="send-orig">Send original only</button>
        </div>` : ''}`;
    if (editable) {
      card.querySelector('[data-act="send-opt"]').onclick = () => {
        const txt = $(`#opt-text-${t.id}`).innerText.trim();
        if (txt && txt !== o.optimized) { t.finalPrompt = txt; t.edited = true; }
        runExecute(t);
      };
      card.querySelector('[data-act="edit"]').onclick = e => {
        const el = $(`#opt-text-${t.id}`);
        const on = el.getAttribute('contenteditable') !== 'true';
        el.setAttribute('contenteditable', on); if (on) el.focus();
        e.target.textContent = on ? 'Done editing' : 'Edit optimised';
      };
      card.querySelector('[data-act="send-orig"]').onclick = () => { t.sentOriginal = true; t.finalPrompt = o.original; runExecute(t); };
    }
    return card;
  }

  const step = (label, cls) => `<div class="step ${cls}"><span class="ic"></span>${label}</div>`;
  const fmtScore = s => s == null ? '—' : `${(+s).toFixed(1)}/10`;

  function execPending(t) {
    const d = document.createElement('div');
    d.className = 'card';
    const two = !t.sentOriginal && !(t.opt && t.opt.unchanged);
    d.innerHTML = `
      <div class="card-head"><div class="title">${icon(ICONS.chat)}Running on ${esc(t.model)}</div></div>
      <div class="steps">
        ${step(two ? 'Sending original and optimised prompts in parallel' : 'Sending prompt', 'active')}
        ${step('Blind-judging both responses', '')}
        ${step('Updating RL policy with reward', '')}
      </div>`;
    return d;
  }

  function errCard(msg) {
    const d = document.createElement('div');
    d.className = 'card err-card';
    d.innerHTML = `<div class="card-head"><div class="title" style="color:var(--bad)">${icon(ICONS.warn)}Something went wrong</div></div><div class="body">${esc(msg)}</div>`;
    return d;
  }

  function responseCard(t) {
    const e = t.exec;
    const d = document.createElement('div');
    d.className = 'card';
    const show = t.tab || 'optimized';
    const r = e[show];
    const two = !e.same_prompt;
    d.innerHTML = `
      ${two ? `<div class="tabs">
        <button class="tab ${show === 'optimized' ? 'active' : ''}" data-tab="optimized">Optimised prompt response</button>
        <button class="tab ${show === 'original' ? 'active' : ''}" data-tab="original">Original prompt response</button>
      </div>` : ''}
      <div class="response">${r.error ? `<div class="err">${esc(r.error)}</div>` : `<div class="md">${md(r.text) || '<p style="color:var(--text-3)">(empty response)</p>'}</div>`}</div>`;
    d.querySelectorAll('[data-tab]').forEach(b => b.onclick = () => { t.tab = b.dataset.tab; refreshTurn(t); });
    return d;
  }

  function pct(before, after, lowerBetter = false) {
    if (before == null || after == null) return { txt: '—', cls: 'flat' };
    if (before === 0 && after === 0) return { txt: '0%', cls: 'flat' };
    const p = before === 0 ? 100 : ((after - before) / Math.abs(before)) * 100;
    const good = lowerBetter ? p < 0 : p > 0;
    const cls = Math.abs(p) < 0.5 ? 'flat' : good ? 'up' : 'down';
    return { txt: `${p > 0 ? '+' : ''}${p.toFixed(0)}%`, cls };
  }

  function metric(title, b, a, { max, unit = '', lowerBetter = false, neutral = false, dp = 1 } = {}) {
    const d = pct(b, a, lowerBetter);
    const cls = neutral ? 'flat' : d.cls;
    const m = max ?? Math.max(b || 0, a || 0, 0.0001);
    const w = v => v == null ? 0 : Math.min(100, (v / m) * 100);
    const f = v => v == null ? '—' : (+v).toFixed(dp) + unit;
    return `
      <div class="metric">
        <div class="mh">${title}<span class="delta ${cls}">${d.txt}</span></div>
        <div class="mrow"><span>Before</span><div class="mbar before"><i data-w="${w(b)}"></i></div><span class="v">${f(b)}</span></div>
        <div class="mrow"><span>After</span><div class="mbar after"><i data-w="${w(a)}"></i></div><span class="v">${f(a)}</span></div>
      </div>`;
  }

  function effCard(t) {
    const e = t.exec;
    const d = document.createElement('div');
    d.className = 'card';
    const so = e.scores.original, sn = e.scores.optimized;
    const ob = so?.overall, oa = sn?.overall;
    const q = pct(ob, oa);
    const ring = oa == null ? 0 : oa * 10;
    const C = 2 * Math.PI * 30;
    let headline, sub;
    if (e.same_prompt) { headline = 'Same prompt sent'; sub = 'The original was already the best candidate, so before and after are identical.'; }
    else if (oa == null) { headline = 'Not scored'; sub = 'The judge could not score one of the responses.'; }
    else { headline = `${q.txt} quality`; sub = `Judge score ${ob.toFixed(1)} → ${oa.toFixed(1)} out of 10`; }

    d.innerHTML = `
      <div class="card-head"><div class="title">${icon(ICONS.chart)}Prompt efficiency: before vs after</div><div class="meta">blind A/B judge</div></div>
      <div class="eff-summary">
        <div class="ring">
          <svg viewBox="0 0 74 74"><defs><linearGradient id="ringGrad" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="hsl(210 95% 65%)"/><stop offset="55%" stop-color="hsl(258 90% 70%)"/><stop offset="100%" stop-color="hsl(330 85% 68%)"/></linearGradient></defs>
            <circle class="track" cx="37" cy="37" r="30"/>
            <circle class="val" cx="37" cy="37" r="30" stroke-dasharray="${C}" stroke-dashoffset="${C}" data-off="${C * (1 - ring / 100)}"/>
          </svg>
          <div class="num">${oa == null ? '—' : oa.toFixed(1)}</div>
        </div>
        <div class="eff-text"><div class="big ${e.same_prompt ? 'flat' : q.cls}">${headline}</div><div class="sub">${sub}</div></div>
      </div>
      <div class="metrics">
        ${metric('Overall quality', ob, oa, { max: 10 })}
        ${['relevance', 'completeness', 'accuracy', 'clarity'].map(k => metric(k, so?.[k], sn?.[k], { max: 10 })).join('')}
        ${metric('Latency', e.original.latency, e.optimized.latency, { unit: 's', lowerBetter: true, dp: 2 })}
        ${metric('Prompt tokens ≈', e.original.prompt_tokens, e.optimized.prompt_tokens, { neutral: true, dp: 0 })}
        ${metric('Response tokens ≈', e.original.response_tokens, e.optimized.response_tokens, { neutral: true, dp: 0 })}
      </div>
      ${e.verdict ? `<div class="verdict">${esc(e.verdict)}</div>` : ''}
      ${e.learning && e.learning.strategy ? `<div class="learn">RL update · <b>${esc(t.opt.strategy_label)}</b> · reward ${e.learning.reward >= 0 ? '+' : ''}${e.learning.reward.toFixed(2)} · Q ${e.learning.old_q.toFixed(2)} → ${e.learning.new_q.toFixed(2)}</div>` : ''}`;
    requestAnimationFrame(() => requestAnimationFrame(() => {
      d.querySelectorAll('.mbar i').forEach(i => i.style.width = i.dataset.w + '%');
      const v = d.querySelector('.ring .val'); if (v) v.style.strokeDashoffset = v.dataset.off;
    }));
    return d;
  }

  // ---------- flow ----------
  async function send(prompt) {
    if (state.busy || !prompt.trim()) return;
    if (!state.model) return toast('Pick a model first');
    if (state.providers[state.provider] && !state.providers[state.provider].has_key) return toast(`${state.provider} API key missing in .env`);

    let chat = current();
    if (!chat) {
      chat = { id: uid(), title: prompt.slice(0, 48), turns: [], created: Date.now() };
      state.chats.unshift(chat); state.currentId = chat.id;
      renderChatList();
    }
    $('#welcome').style.display = 'none';
    const turn = { id: uid(), prompt, provider: state.provider, model: state.model, status: 'optimizing' };
    chat.turns.push(turn);
    refreshTurn(turn); scrollDown();
    setBusy(true);

    try {
      turn.opt = await api('/api/optimize', { prompt, provider: turn.provider, model: turn.model });
      renderPolicy(turn.opt.policy);
    } catch (e) {
      turn.status = 'error'; turn.error = e.message; refreshTurn(turn); setBusy(false); return;
    }

    if (state.review && !turn.opt.unchanged) {
      turn.status = 'review'; refreshTurn(turn); scrollDown(); setBusy(false);
      return;  // waits for the user to press a button in the card
    }
    runExecute(turn);
  }

  async function runExecute(turn) {
    setBusy(true);
    turn.status = 'executing';
    refreshTurn(turn); scrollDown();
    const optimized = turn.finalPrompt ?? turn.opt.optimized;
    const original = turn.sentOriginal ? optimized : turn.opt.original;
    const strategy = turn.sentOriginal || turn.edited ? null : turn.opt.strategy;
    try {
      turn.exec = await api('/api/execute', { original, optimized, provider: turn.provider, model: turn.model, strategy });
      turn.status = 'done';
      renderPolicy(turn.exec.policy, turn.exec.learning?.strategy);
    } catch (e) {
      turn.status = 'error'; turn.error = e.message;
    }
    refreshTurn(turn); scrollDown(); setBusy(false);
  }

  function setBusy(b) {
    state.busy = b;
    updateSend();
    $('#prompt-input').placeholder = b ? 'Working…' : 'Ask anything — it will be RL-optimised first…';
  }
  function updateSend() { $('#send-btn').disabled = state.busy || !$('#prompt-input').value.trim(); }

  // ---------- events ----------
  function bind() {
    const input = $('#prompt-input');
    const autosize = () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 200) + 'px'; };
    input.addEventListener('input', () => { autosize(); updateSend(); });
    input.addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); $('#composer').requestSubmit(); }
    });
    $('#composer').addEventListener('submit', e => {
      e.preventDefault();
      const v = input.value; if (!v.trim() || state.busy) return;
      input.value = ''; autosize(); updateSend(); send(v.trim());
    });

    document.querySelectorAll('.provider-btn').forEach(b => b.onclick = () => { if (!state.busy) setProvider(b.dataset.provider); });
    $('#model-select').onchange = e => { state.model = e.target.value; savePrefs(); };
    $('#review-toggle').checked = state.review = !!prefs.review;
    $('#review-toggle').onchange = e => { state.review = e.target.checked; savePrefs(); };

    $('#chips').addEventListener('click', e => { const c = e.target.closest('.chip'); if (c) send(c.textContent.trim()); });
    $('#new-chat-btn').onclick = newChat;
    $('#chat-list').addEventListener('click', e => {
      const del = e.target.closest('[data-del]');
      if (del) {
        e.stopPropagation();
        if (state.busy) return;
        state.chats = state.chats.filter(c => c.id !== del.dataset.del);
        if (state.currentId === del.dataset.del) state.currentId = null;
        save(); renderChatList(); renderMessages(); return;
      }
      const item = e.target.closest('.chat-item'); if (item && !state.busy) openChat(item.dataset.id);
    });
    $('#reset-policy-btn').onclick = async () => {
      if (!confirm('Reset all learned strategy values to zero?')) return;
      try { renderPolicy(await api('/api/policy/reset', {})); toast('RL policy reset'); } catch (e) { toast(e.message); }
    };
    $('#collapse-sidebar').onclick = () => $('#sidebar').classList.add('collapsed');
    $('#open-sidebar').onclick = () => $('#sidebar').classList.toggle('collapsed');
    window.addEventListener('resize', moveGlider);
  }

  // Turns left mid-flight from a previous session can't resume.
  state.chats.forEach(c => c.turns.forEach(t => {
    if (['optimizing', 'executing'].includes(t.status)) { t.status = 'error'; t.error = 'Interrupted (page was reloaded).'; }
    if (t.status === 'review') t.status = 'error', t.error = 'Review was not completed before the page reloaded.';
  }));

  bind();
  if (window.innerWidth <= 860) $('#sidebar').classList.add('collapsed');
  renderChatList();
  renderMessages();
  initModels();
  loadPolicy();
  document.fonts?.ready.then(moveGlider);
})();
