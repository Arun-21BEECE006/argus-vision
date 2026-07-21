/* Argus Vision — shared front-end helpers */

function setupUpload(){
  const drop  = document.getElementById('drop');
  const input = document.getElementById('file');
  const fname = document.getElementById('fname');
  const conf  = document.getElementById('conf');
  const confV = document.getElementById('confVal');

  if (conf && confV){
    conf.addEventListener('input', () => confV.textContent = (+conf.value).toFixed(2));
  }
  if (!drop) return;

  const pick = (file) => {
    window.__file = file;
    if (fname) fname.textContent = file ? file.name : '';
  };
  drop.addEventListener('click', () => input.click());
  input.addEventListener('change', () => pick(input.files[0]));
  ['dragover','dragenter'].forEach(ev =>
    drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('drag'); }));
  ['dragleave','drop'].forEach(ev =>
    drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('drag'); }));
  drop.addEventListener('drop', e => { if (e.dataTransfer.files[0]) pick(e.dataTransfer.files[0]); });
}

function toggleLoad(on){
  const l = document.getElementById('load');
  if (l) l.classList.toggle('on', on);
}

function showErr(msg){
  const e = document.getElementById('err');
  if (e) e.textContent = msg || '';
}

function renderReadout(d){
  const wrap  = document.getElementById('readout');
  const chips = document.getElementById('chips');
  const list  = document.getElementById('detlist');
  if (!wrap) return;
  wrap.style.display = 'block';

  let chipHtml = `<span class="chip total">TOTAL ${d.total}</span>`;
  const entries = Object.entries(d.summary || {}).sort((a,b) => b[1]-a[1]);
  for (const [k,v] of entries) chipHtml += `<span class="chip">${k} <b>${v}</b></span>`;
  chips.innerHTML = chipHtml;

  if (Array.isArray(d.details)){
    list.style.display = 'block';
    list.innerHTML = d.details
      .sort((a,b) => b.conf - a.conf)
      .map(x => `<div class="row"><span class="lbl">${x.label}</span><span class="cf">${(x.conf*100).toFixed(1)}%</span></div>`)
      .join('') || '<div class="row"><span>No objects detected.</span></div>';
  } else if (list){
    list.style.display = 'none';
  }
}
