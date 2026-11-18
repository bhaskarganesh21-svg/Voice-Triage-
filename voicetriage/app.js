const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const LOCALE={hi:'hi-IN',en:'en-IN',mr:'mr-IN',ta:'ta-IN',te:'te-IN',bn:'bn-IN'};
let tab='voice',demoId=null,rec=null,recording=false,spoken='';
const j=async(u,o)=>{const r=await fetch(u,o);if(!r.ok)throw new Error((await r.text()).slice(0,200)||r.status);return r.json()};
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// Tabs
$$('.tabs button').forEach(b=>b.onclick=()=>{tab=b.dataset.tab;$$('.tabs button').forEach(x=>x.classList.toggle('on',x===b));
  $$('.pane').forEach(p=>p.classList.toggle('on',p.id==='p-'+tab))});

// Health + demo cases
j('/api/health').then(h=>{const p=$('#health');p.classList.add('ok');p.innerHTML='<i></i>Live'+(h.whisper_model?' · '+h.whisper_model:'')}).catch(()=>$('#health').innerHTML='<i></i>Offline');
j('/api/cases').then(d=>{const l=Array.isArray(d)?d:(d.cases||[]);
  $('#cases').innerHTML=l.map(c=>{const id=c.case_id||c.id;return `<button class="case" data-id="${esc(id)}">${esc(id)}<small>${esc(c.language_name||c.language||'')} ${c.ground_truth?'· '+esc(c.ground_truth.urgency):''}</small></button>`}).join('')||'No demo cases found';
  $$('.case').forEach(b=>b.onclick=()=>{demoId=b.dataset.id;$$('.case').forEach(x=>x.classList.toggle('sel',x===b))})}).catch(()=>$('#cases').textContent='Could not load cases');

// Live voice (browser Web Speech API — needs HTTPS, which the cloud URL provides)
const SR=window.SpeechRecognition||window.webkitSpeechRecognition;
$('#mic').onclick=()=>{
  if(!SR){$('#micHint').textContent='Voice not supported in this browser — use Chrome, or the Type tab.';return}
  if(recording){rec.stop();return}
  rec=new SR();rec.lang=LOCALE[$('#lang').value];rec.continuous=true;rec.interimResults=true;spoken='';
  rec.onresult=e=>{let t='';for(const r of e.results)t+=r[0].transcript+' ';spoken=t.trim();$('#live').textContent=spoken};
  rec.onstart=()=>{recording=true;$('#mic').classList.add('rec');$('#micHint').textContent='Listening… tap again to stop'};
  rec.onend=()=>{recording=false;$('#mic').classList.remove('rec');$('#micHint').textContent='Tap the mic to speak again'};
  rec.onerror=e=>$('#micHint').textContent='Mic error: '+e.error;
  rec.start()};

// Run triage
$('#run').onclick=async()=>{
  const lang=$('#lang').value;let req;
  if(tab==='demo'){if(!demoId)return alert('Pick a demo case first');req=()=>j('/api/demo/'+encodeURIComponent(demoId))}
  else{const text=(tab==='voice'?spoken:$('#txt').value).trim();if(!text)return alert('Please speak or type the symptoms first');
    req=()=>j('/api/triage/text',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({text,transcript:text,language:lang,language_hint:lang})})}
  const btn=$('#run');btn.disabled=true;$('#result').innerHTML='<div class="loader"><div class="spin"></div>Analysing symptoms…</div>';
  try{render(await req())}catch(e){$('#result').innerHTML=`<div class="empty"><div class="big">⚠️</div><h2>Something went wrong</h2><p>${esc(e.message)}</p></div>`}
  btn.disabled=false};

const chips=(a,c='')=>(a&&a.length?a:['—']).map(x=>`<span class="chip ${c}">${esc(x)}</span>`).join('');
function render(d){
  const r=d.result||d,u=r.urgency_level||'ROUTINE',conf=Math.round((r.confidence_score||0)*100);
  $('#result').innerHTML=`
  <div class="banner ${u}"><small>Urgency</small><h2>${u}</h2><div>Refer to <b>${esc(r.suggested_specialty)}</b></div></div>
  <div class="grid">
    <div class="stat"><small>Confidence</small><b>${conf}%</b><div class="bar"><span style="width:${conf}%"></span></div></div>
    <div class="stat"><small>Severity</small><b>${esc(r.severity)}</b></div>
    <div class="stat"><small>Duration</small><b>${esc(r.duration)}</b></div>
    <div class="stat"><small>Age / Gender</small><b>${esc(r.age||'N/A')} · ${esc(r.gender)}</b></div>
  </div>
  <div class="sec"><h4>Symptoms</h4><div class="chips">${chips(r.symptoms)}</div></div>
  ${r.red_flags_triggered?.length?`<div class="sec"><h4>Red flags</h4><div class="chips">${chips(r.red_flags_triggered,'flag')}</div></div>`:''}
  ${r.existing_diseases?.length?`<div class="sec"><h4>Existing conditions</h4><div class="chips">${chips(r.existing_diseases)}</div></div>`:''}
  <div class="sec"><h4>Why this result</h4><p>${esc(r.explanation)}</p></div>
  <div class="sec"><h4>Transcript · ${esc(r.detected_language)}</h4><p>${esc(r.transcript)}</p></div>
  <div class="disc">Decision support only — not a diagnosis. Completed in ${esc(r.total_time_seconds)}s.</div>`}

// Drawers
$$('[data-drawer]').forEach(b=>b.onclick=async()=>{
  const k=b.dataset.drawer;$('#dTitle').textContent=k==='stats'?'Statistics':'Recent triage';$('#dBody').textContent='Loading…';$('#drawer').classList.add('open');
  try{const d=await j('/api/'+k);
    if(k==='history'){const l=Array.isArray(d)?d:(d.history||d.results||[]);
      $('#dBody').innerHTML=l.map(r=>`<div class="row"><b>${esc(r.urgency_level)}</b> · ${esc(r.suggested_specialty)}<small>${esc(r.timestamp)} · ${esc((r.transcript||'').slice(0,70))}</small></div>`).join('')||'No history yet'}
    else $('#dBody').innerHTML=Object.entries(d.stats||d).map(([a,v])=>`<div class="row"><small>${esc(a.replace(/_/g,' '))}</small>${esc(typeof v==='object'?JSON.stringify(v):v)}</div>`).join('')}
  catch(e){$('#dBody').textContent='Unavailable: '+e.message}});
$('#dClose').onclick=()=>$('#drawer').classList.remove('open');
