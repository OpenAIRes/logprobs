'use client';

import { useEffect, useMemo, useState } from 'react';

const API = 'http://127.0.0.1:8899';
type LocalRequestInit = RequestInit & { targetAddressSpace?: 'loopback' };
const localFetch = (path: string, init: RequestInit = {}) =>
  fetch(`${API}${path}`, { ...init, mode: 'cors', targetAddressSpace: 'loopback' } as LocalRequestInit);

/* Loaded by layout.tsx from server.py, not bundled: one policy and one dialog
   for every part of the package. See ask-policy.js. */
type AskVerdict = 'yes' | 'all' | 'skip' | 'stop';
declare global {
  interface Window {
    AskPolicy?: {
      guard: (request: unknown, info: { total: number }) => Promise<AskVerdict | boolean>;
      load: () => Promise<unknown>;
      describe: () => string;
      logprobsFor: (model: string) => number;
    };
    CallReport?: { report: (stage: string, detail?: string) => void };
    ASK_POLICY_BASE?: string;
  }
}
/* These `alternatives` numbers are the API's own cap, and the authority for them
   is AskPolicy.logprobsFor in ask-policy.js -- kept literal here only because
   this table also carries the labels and notes the UI prints next to them. If
   they ever disagree, that module is right. */
const MODELS = {
  'gpt-3.5-turbo-instruct': { alternatives: 20, label: 'GPT‑3.5 Turbo Instruct', note: 'Legacy Completions · až 20 alternativ tokenu' },
  'davinci-002': { alternatives: 5, label: 'Davinci 002', note: 'Legacy Completions · nejvýše 5 alternativ' },
  'babbage-002': { alternatives: 5, label: 'Babbage 002', note: 'Legacy Completions · nejvýše 5 alternativ' },
} as const;
type Model = keyof typeof MODELS;
type Tab = 'playground' | 'logs' | 'explore';
type ParameterDefaults = { temperature:number; maxTokens:number; topP:number; logprobs:number; frequency:number; presence:number };

const STUDIO_DEFAULTS: Record<Model,ParameterDefaults> = {
  'gpt-3.5-turbo-instruct': { temperature:0, maxTokens:20, topP:1, logprobs:20, frequency:0, presence:0 },
  'davinci-002': { temperature:0, maxTokens:5, topP:1, logprobs:5, frequency:0, presence:0 },
  'babbage-002': { temperature:0, maxTokens:5, topP:1, logprobs:5, frequency:0, presence:0 },
};
const OPENAI_DEFAULTS: ParameterDefaults = { temperature:1, maxTokens:16, topP:1, logprobs:0, frequency:0, presence:0 };
const INITIAL_DEFAULTS = Object.fromEntries(Object.keys(MODELS).map(model=>[model,{...OPENAI_DEFAULTS}])) as Record<Model,ParameterDefaults>;
const DEFAULTS_STORAGE_KEY = 'prompt-studio-parameter-defaults-v1';

type LogRecord = {
  id?: string; model?: string; created?: number;
  request?: { prompt?: string; max_tokens?: number; temperature?: number };
  choices?: Array<{ text?: string; finish_reason?: string; logprobs?: { tokens?: string[]; token_logprobs?: number[] } }>;
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
};

type ApiResult = LogRecord & {error?:string; detail?:string; saved?:boolean; from_cache?:boolean; api_completed?:boolean; record?:LogRecord; hit?:boolean; exact?:boolean};

const nav = [['⌂','Playground','playground'],['≡','Logs','logs'],['⌘','Logprobs Explorer','explore'],['⌁','Greedy paths','explore'],['◇','Alternatives','explore'],['⚙','Settings','playground']] as const;

export default function Home() {
  const [tab,setTab]=useState<Tab>('playground');
  const [model,setModel]=useState<Model>('gpt-3.5-turbo-instruct');
  const [prompt,setPrompt]=useState(''); const [output,setOutput]=useState('');
  const [temperature,setTemperature]=useState(OPENAI_DEFAULTS.temperature); const [maxTokens,setMaxTokens]=useState<number>(OPENAI_DEFAULTS.maxTokens);
  const [topP,setTopP]=useState(1); const [frequency,setFrequency]=useState(0); const [presence,setPresence]=useState(0);
  const [logprobs,setLogprobs]=useState(OPENAI_DEFAULTS.logprobs); const [live,setLive]=useState(true); const [busy,setBusy]=useState(false);
  const [runStatus,setRunStatus]=useState('');
  const [unsaved,setUnsaved]=useState<LogRecord|null>(null);
  const [parameterDefaults,setParameterDefaults]=useState<Record<Model,ParameterDefaults>>(INITIAL_DEFAULTS);
  const [connected,setConnected]=useState(false); const [stats,setStats]=useState<Record<string,unknown>|null>(null);
  const [connectionHint,setConnectionHint]=useState('Spusťte server.py');
  const [logs,setLogs]=useState<LogRecord[]>([]); const [query,setQuery]=useState(''); const [selected,setSelected]=useState<LogRecord|null>(null);
  const cap=MODELS[model];

  useEffect(()=>{
    const timer=window.setTimeout(()=>{
      try{
        const saved=JSON.parse(localStorage.getItem(DEFAULTS_STORAGE_KEY)||'{}') as Partial<Record<Model,Partial<ParameterDefaults>>>;
        const merged=Object.fromEntries((Object.keys(MODELS) as Model[]).map(key=>[key,{...OPENAI_DEFAULTS,...saved[key]}])) as Record<Model,ParameterDefaults>;
        const initial=merged['gpt-3.5-turbo-instruct']; setParameterDefaults(merged);
        setTemperature(initial.temperature); setMaxTokens(initial.maxTokens); setTopP(initial.topP); setLogprobs(initial.logprobs); setFrequency(initial.frequency); setPresence(initial.presence);
      }catch{ /* Keep the built-in OpenAI defaults. */ }
    },0);
    void refreshLogs();
    return ()=>window.clearTimeout(timer);
  },[]);

  function applyDefaults(values:ParameterDefaults,targetModel:Model=model){
    setTemperature(values.temperature); setMaxTokens(values.maxTokens); setTopP(values.topP);
    setLogprobs(Math.min(values.logprobs,MODELS[targetModel].alternatives)); setFrequency(values.frequency); setPresence(values.presence);
  }
  function selectModel(next:Model){setModel(next);applyDefaults(parameterDefaults[next],next)}
  function currentDefaults():ParameterDefaults{return {temperature,maxTokens,topP,logprobs,frequency,presence}}
  function saveDefaults(values:ParameterDefaults=currentDefaults()){
    const next={...parameterDefaults,[model]:values}; setParameterDefaults(next);
    localStorage.setItem(DEFAULTS_STORAGE_KEY,JSON.stringify(next));
  }
  function saveOne(key:keyof ParameterDefaults,value:number){saveDefaults({...parameterDefaults[model],[key]:value})}
  function restoreStudioDefaults(){
    const next={...parameterDefaults,[model]:STUDIO_DEFAULTS[model]}; setParameterDefaults(next);
    localStorage.setItem(DEFAULTS_STORAGE_KEY,JSON.stringify(next)); applyDefaults(STUDIO_DEFAULTS[model]);
  }

  // The policy is stored on the server, so refresh it once the page is up.
  useEffect(()=>{void window.AskPolicy?.load()},[]);

  async function refreshLogs(){
    try{
      const [s,q]=await Promise.all([localFetch('/api/stats').then(r=>{if(!r.ok)throw new Error(String(r.status));return r.json() as Promise<Record<string,unknown>>}),localFetch('/api/records?limit=200').then(r=>{if(!r.ok)throw new Error(String(r.status));return r.json() as Promise<{records:LogRecord[]}>})]);
      setStats(s); setLogs(q.records||[]); setConnected(true); setConnectionHint('127.0.0.1:8899');
    }catch{setConnected(false);setConnectionHint(location.protocol==='https:'?'Povolte této stránce přístup k místní síti':'Spusťte server.py')}
  }

  async function run(){
    if(busy||unsaved)return; setBusy(true); setOutput(''); setSelected(null); setRunStatus(temperature===0?'Hledám shodu v historii…':'Připravuji API volání…');
    const body={prompt,model,max_tokens:maxTokens,temperature,top_p:topP,frequency_penalty:frequency,presence_penalty:presence,logprobs};
    try{
      if(temperature===0){
      const lookup=await localFetch('/api/lookup',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,exact:true})}).then(async r=>{const data=await r.json() as ApiResult;if(!r.ok)throw new Error(data.error||'Vyhledání v historii selhalo');return data});
      if(lookup.exact!==true)throw new Error('Server nepodporuje bezpečné ukládání. Restartujte aktualizovaný server.py.');
      if(lookup.hit&&lookup.record){setSelected(lookup.record);setRunStatus('From history');setOutput(lookup.record?.choices?.[0]?.text||'Uložený záznam nemá textový výstup.');return}
      }
      if(!live){setRunStatus('API nebylo voláno');setOutput('API režim je vypnutý, takže nebylo provedeno žádné placené volání.');return}
      /* One rule for the whole package, and it fails closed: with the policy
         script missing there is nothing to ask with, so nothing is sent. The
         request is built once and both shown and posted, so the dialog cannot
         describe something other than what goes out. */
      if(!window.AskPolicy){setRunStatus('Nelze volat');setOutput('Politika volání se nenačetla z http://127.0.0.1:8899 (ask-policy.js). Spusťte server.py a načtěte stránku znovu — bez ní se placené volání neprovede.');return}
      const request={...body,confirmed:true};
      window.CallReport?.report('asking',`${model} · ${maxTokens} tokenů`);
      const verdict=await window.AskPolicy.guard(request,{total:1});
      if(verdict!=='yes'&&verdict!=='all'&&verdict!==true){window.CallReport?.report('declined');setRunStatus('API nebylo voláno');setOutput('Volání zrušeno — nic se neposlalo.');return}
      window.CallReport?.report('calling',`${model} · ${maxTokens} tokenů`);
      setRunStatus('Volám API…');
      const res=await localFetch('/api/complete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
      const data=await res.json() as ApiResult;
      if(data.api_completed && data.record){setSelected(data.record);setUnsaved(data.record);setOutput(data.record.choices?.[0]?.text??'Prázdná odpověď.');setRunStatus(data.error||'Odpověď není uložená');return}
      if(!res.ok)throw new Error([data.error||'API chyba',data.detail].filter(Boolean).join('\n'));
      if(data.saved!==true){setSelected(data);setUnsaved(data);setOutput(data.choices?.[0]?.text??'Prázdná odpověď.');setRunStatus('Server nepotvrdil uložení. Stáhněte odpověď a aktualizujte server.');return}
      window.CallReport?.report(data.from_cache?'cached':'saved',model);
      setSelected(data);setRunStatus(data.from_cache?'Načteno z historie':'Odpověď uložena do historie');setOutput(data.choices?.[0]?.text??'Prázdná odpověď.');await refreshLogs();
    }catch(e){const m=e instanceof Error?e.message:'Lokální server není dostupný.';window.CallReport?.report(/nedosa|unreachable|getaddrinfo|Failed to fetch/i.test(m)?'unreachable':'failed',m);setRunStatus('Chyba');setOutput(m)}finally{setBusy(false)}
  }

  function downloadUnsaved(){
    if(!unsaved)return;
    const url=URL.createObjectURL(new Blob([JSON.stringify(unsaved,null,2)],{type:'application/json'}));
    const link=document.createElement('a');link.href=url;link.download=`completion-${unsaved.id||'unsaved'}.json`;link.click();
    window.setTimeout(()=>URL.revokeObjectURL(url),1000);
  }

  const shown=useMemo(()=>logs.filter(r=>JSON.stringify(r).toLowerCase().includes(query.toLowerCase())),[logs,query]);

  return <main className="app-shell">
    <aside className="sidebar"><div className="project"><div className="brandmark">P</div><div><b>Prompt Studio</b><small>GPT workspace</small></div><span>⌃</span></div><label className="search">⌕<input placeholder="Search"/></label><nav>{nav.map(([icon,label,target])=><button key={label} className={tab===target?'active':''} onClick={()=>setTab(target)}><span>{icon}</span>{label}</button>)}</nav><button className="connection" onClick={()=>void refreshLogs()}><i className={connected?'ok':''}/><div><b>{connected?'GPT data connected':'Connect local data'}</b><small>{connectionHint}</small></div></button><div className="profile"><span>J</span><div>Jan<small>Local workspace</small></div></div></aside>
    <section className="workspace"><header className="topbar"><div><h1>{tab==='playground'?'Prompts':tab==='logs'?'Saved logs':'Logprobs Explorer'}</h1><span className="crumb">gpt / {tab}</span></div><div className="header-actions"><button onClick={refreshLogs}>↻ Refresh</button><button className="ghost">↝ Compare</button><button className="dark">‹/› Code</button></div></header>
      {tab==='playground'?<div className="studio">
        <section className="controls"><Section title="Prompt"><textarea className="prompt" value={prompt} onChange={e=>setPrompt(e.target.value)} placeholder="Enter a prompt for the model…"/><button className="add">＋ Add message</button></Section><Section title="Model"><Select label="Model" value={model} onChange={v=>selectModel(v as Model)} options={Object.entries(MODELS).map(([v,c])=>[v,c.label])}/><p className="capability">{cap.note}</p><Select label="Response format" value="text" options={[["text","Text"]]}/><Slider label="Temperature" value={temperature} set={setTemperature} min={0} max={2} step={.01} defaultValue={parameterDefaults[model].temperature} onDefault={()=>saveOne('temperature',temperature)}/><Slider label="Max tokens" value={maxTokens} set={setMaxTokens} min={1} max={4096} step={1} defaultValue={parameterDefaults[model].maxTokens} onDefault={()=>saveOne('maxTokens',maxTokens)}/><Slider label="Top P" value={topP} set={setTopP} min={0} max={1} step={.01} defaultValue={parameterDefaults[model].topP} onDefault={()=>saveOne('topP',topP)}/><Slider label="Top logprobs" value={logprobs} set={setLogprobs} min={0} max={cap.alternatives} step={1} defaultValue={parameterDefaults[model].logprobs} onDefault={()=>saveOne('logprobs',logprobs)}/><Slider label="Frequency penalty" value={frequency} set={setFrequency} min={-2} max={2} step={.01} defaultValue={parameterDefaults[model].frequency} onDefault={()=>saveOne('frequency',frequency)}/><Slider label="Presence penalty" value={presence} set={setPresence} min={-2} max={2} step={.01} defaultValue={parameterDefaults[model].presence} onDefault={()=>saveOne('presence',presence)}/><div className="defaults-panel"><div><b>Výchozí hodnoty</b><small>Samostatně pro každý model · uloženo v tomto prohlížeči</small></div><div><button onClick={()=>saveDefaults()}>Uložit vše</button><button onClick={()=>applyDefaults(parameterDefaults[model])}>Obnovit</button><button onClick={()=>applyDefaults(OPENAI_DEFAULTS)} title="temperature 1 · max tokens 16 · top p 1 · logprobs vypnuto · penalties 0">Použít OpenAI</button><button onClick={restoreStudioDefaults}>Studio</button></div></div><Toggle label="Enable paid API calls" on={live} set={setLive}/></Section></section>
        <section className="conversation">{runStatus&&<div className="run-status" aria-live="polite"><i className={busy?"working":""}/>{runStatus}{unsaved&&<><button onClick={downloadUnsaved}>Stáhnout neuloženou odpověď</button><button onClick={()=>{if(window.confirm("Máte odpověď staženou? Další odeslání může znamenat nové placené volání.")){setUnsaved(null);setRunStatus('')}}}>Pokračovat</button></>}</div>}{output?<div className="response"><div className="response-head"><span>{model}</span></div><pre>{output}</pre>{selected&&<TokenStrip record={selected}/>}</div>:<div className="empty"><span>☵</span><b>Your conversation will appear here</b><p>Nejprve se vždy hledá přesná shoda v lokálních záznamech.</p></div>}<div className="composer"><textarea value={prompt} onChange={e=>setPrompt(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();void run()}}} placeholder="Ask anything"/><button>＋</button><button className="send" onClick={()=>void run()} disabled={busy||!!unsaved}>{busy?'…':'↑'}</button></div></section>
      </div>:tab==='logs'?<div className="logs-page"><div className="stats"><Stat label="Records" value={num(stats?.records)}/><Stat label="Distinct prompts" value={num(stats?.distinct_prompts)}/><Stat label="Prompt tokens" value={num(stats?.prompt_tokens)}/><Stat label="Completion tokens" value={num(stats?.completion_tokens)}/></div><div className="logs-tools"><label className="log-search">⌕<input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search prompts, outputs, IDs…"/></label><span>{shown.length} loaded</span></div><div className="log-layout"><div className="log-list">{shown.map((r,i)=><button key={r.id||i} className={selected===r?'selected':''} onClick={()=>setSelected(r)}><div><b>{r.model||'unknown model'}</b><span>{r.choices?.[0]?.finish_reason||'record'}</span></div><p>{r.request?.prompt||'(empty prompt)'}</p><small>{(r.choices?.[0]?.text||'').slice(0,120)}</small></button>)}</div><div className="detail">{selected?<><div className="detail-head"><div><b>{selected.model}</b><small>{selected.id}</small></div><span>{selected.usage?.total_tokens||0} tokens</span></div><h3>Prompt</h3><pre>{selected.request?.prompt||'(empty)'}</pre><h3>Completion</h3><pre>{selected.choices?.[0]?.text||'(empty)'}</pre><TokenStrip record={selected}/></>:<div className="detail-empty">Select a saved log</div>}</div></div></div>:<Explorer connected={connected}/>} </section>
  </main>
}

function Section({title,children}:{title:string;children:React.ReactNode}){return <div className="section"><div className="section-title"><b>{title}</b><span>⌃</span></div>{children}</div>}
function Select({label,value,onChange,options}:{label:string;value:string;onChange?:(v:string)=>void;options:string[][]}){return <label className="field"><span>{label}</span><select value={value} onChange={e=>onChange?.(e.target.value)}>{options.map(([v,l])=><option value={v} key={v}>{l}</option>)}</select></label>}
function Slider({label,value,set,min,max,step,defaultValue,onDefault}:{label:string;value:number;set:(v:number)=>void;min:number;max:number;step:number;defaultValue:number;onDefault:()=>void}){const same=Math.abs(value-defaultValue)<.0001;return <label className="slider"><span><span>{label}</span><span className="slider-meta"><button type="button" onClick={e=>{e.preventDefault();set(defaultValue)}} disabled={same} title={`Obnovit výchozí hodnotu ${defaultValue}`}>↺</button><button type="button" className={same?'is-default':''} onClick={e=>{e.preventDefault();onDefault()}} title={same?'Toto je výchozí hodnota':'Nastavit tuto hodnotu jako výchozí'}>◇</button><b>{Number.isInteger(step)?value:value.toFixed(2)}</b></span></span><input type="range" value={value} onChange={e=>set(Number(e.target.value))} min={min} max={max} step={step}/></label>}
function Toggle({label,on,set}:{label:string;on:boolean;set:(v:boolean)=>void}){return <div className="toggle-row"><span>{label}</span><button className={on?'toggle on':'toggle'} onClick={()=>set(!on)}><i/></button></div>}
function Stat({label,value}:{label:string;value:string}){return <div><span>{label}</span><b>{value}</b></div>}
function num(v:unknown){return typeof v==='number'?v.toLocaleString('cs-CZ'):'—'}
function TokenStrip({record}:{record:LogRecord}){const lp=record.choices?.[0]?.logprobs;const tokens=lp?.tokens||[];const [limit,setLimit]=useState('all');const visible=limit==='all'?tokens:tokens.slice(0,Number(limit));return <div className="token-block"><div className="token-toolbar"><span>{visible.length.toLocaleString('cs-CZ')} / {tokens.length.toLocaleString('cs-CZ')} tokens</span><label>Zobrazit <select value={limit} onChange={e=>setLimit(e.target.value)}><option value="all">maximum</option><option value="80">80</option><option value="250">250</option><option value="1000">1 000</option></select></label></div><div className="tokens">{visible.map((t,i)=><span key={i} title={`logprob ${lp?.token_logprobs?.[i]??'—'}`} style={{opacity:Math.max(.35,1+Number(lp?.token_logprobs?.[i]||0)/10)}}>{t.replaceAll(' ','·')}</span>)}</div></div>}
function Explorer({connected}:{connected:boolean}){return <div className="explorer"><div className="explorer-card"><span className="big-icon">⌘</span><h2>Logprobs Explorer</h2><p>Pokročilé pohledy z původního projektu zůstávají dostupné se stejnými daty.</p><div className="view-grid"><a href={`${API}/`} target="_blank">Greedy path · maximum <span>→</span></a><a href={`${API}/logprobs.html?view=completions&top=200`} target="_blank">Completions <span>→</span></a><a href={`${API}/logprobs.html?view=prefixes&top=200`} target="_blank">Best prefixes <span>→</span></a><a href={`${API}/index.html`} target="_blank">All analysis tools <span>→</span></a></div><small className={connected?'online':'offline'}>{connected?'● Local data server connected':'● Start server.py to open these tools'}</small></div></div>}
