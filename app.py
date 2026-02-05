# -*- coding: utf-8 -*-
"""
BEYOND PRICE AND TIME — PROOF ENGINE v6.1
Copyright © 2026 Truth Communications LLC. All Rights Reserved.

Same as v6 with improved chart label spacing.
Sequential single-file upload for reliable 150-file support.

Requirements:
    pip install flask numpy scipy
"""

import os, io, csv, sys, json, random, traceback, gc, time, uuid
import numpy as np
from scipy import stats
from collections import defaultdict
from datetime import datetime, timedelta
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

SESSIONS = {}
SESSION_TIMEOUT = 600

def cleanup_old_sessions():
    now = time.time()
    for k in [k for k,v in SESSIONS.items() if now - v.get('created',0) > SESSION_TIMEOUT]:
        del SESSIONS[k]

DATE_FORMATS = [
    '%Y-%m-%d','%m/%d/%Y','%d/%m/%Y','%Y/%m/%d','%m-%d-%Y','%d-%m-%Y',
    '%Y-%m-%d %H:%M:%S','%Y-%m-%dT%H:%M:%S','%Y-%m-%dT%H:%M:%SZ',
    '%Y-%m-%dT%H:%M:%S.%f','%Y-%m-%dT%H:%M:%S.%fZ','%Y-%m-%d %H:%M:%S.%f',
    '%m/%d/%Y %H:%M:%S','%m/%d/%Y %H:%M','%m/%d/%y','%d/%m/%y','%Y%m%d',
    '%m-%d-%y','%d-%b-%Y','%d-%b-%y','%b %d, %Y','%B %d, %Y',
    '%d %b %Y','%d %B %Y','%Y.%m.%d','%m.%d.%Y','%d.%m.%Y',
    '%m/%d/%Y %I:%M:%S %p','%m/%d/%Y %I:%M %p',
]
PRICE_ALIASES = {'close','adj close','adj_close','adjclose','adjusted_close','price','last','last_price','lastprice','close_price','closeprice','settlement','settle','closing_price','value','rate','mid','midpoint','vwap','c'}
DATE_ALIASES = {'date','time','datetime','timestamp','ts','period','trade_date','tradedate','day','dt','created_at','t'}
OHLV_ALIASES = {'open','high','low','volume','vol','o','h','l','v','bid','ask','spread','change','pct_change'}
SKIP_ALIASES = {'symbol','ticker','name','exchange','currency','type','sector','industry','country','index','id','row'}

def try_parse_date(s):
    s = s.strip().strip('"').strip("'")
    if not s or s in ('-','--','N/A','null','None',''): return None
    try:
        val = float(s)
        if val > 1e12: return datetime.fromtimestamp(val/1000)
        elif val > 1e9: return datetime.fromtimestamp(val)
    except: pass
    for fmt in DATE_FORMATS:
        try: return datetime.strptime(s, fmt)
        except: continue
    return None

def try_parse_float(s):
    if not s: return None
    s = s.strip().strip('"').strip("'")
    if not s or s in ('-','--','N/A','null','None','','#N/A'): return None
    s = s.replace('$','').replace('\u20ac','').replace('\u00a3','').replace('\u00a5','')
    if ',' in s and '.' in s:
        if s.rindex(',') > s.rindex('.'): s = s.replace('.','').replace(',','.')
        else: s = s.replace(',','')
    elif ',' in s and '.' not in s:
        parts = s.split(',')
        if len(parts)==2 and len(parts[1])!=3: s = s.replace(',','.')
        else: s = s.replace(',','')
    try:
        val = float(s)
        if np.isfinite(val): return val
    except: pass
    return None

def detect_delimiter(content):
    lines = content.split('\n')[:5]
    first = lines[0] if lines else ''
    scores = {'\t':first.count('\t'),',':first.count(','),';':first.count(';'),'|':first.count('|')}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else ','

def identify_columns(header, sample_rows):
    n = len(header)
    hl = [h.strip().lower().replace('\ufeff','').replace(' ','_') for h in header]
    hns = [h.replace('_','').replace('-','').replace(' ','') for h in hl]
    pc = dc = None; sym = "DATA"
    for i,(c,cn) in enumerate(zip(hl,hns)):
        if c in PRICE_ALIASES or cn in PRICE_ALIASES: pc = i
        if c in DATE_ALIASES or cn in DATE_ALIASES: dc = i
        if c in ('symbol','ticker','name') and sample_rows:
            try: sym = sample_rows[0][i].strip().upper()[:10]
            except: pass
    if pc is None or dc is None:
        ds = [0]*n; fs = [0]*n
        for row in sample_rows[:20]:
            for i in range(min(len(row),n)):
                v = row[i].strip() if i<len(row) else ''
                if not v: continue
                if try_parse_date(v): ds[i]+=1
                if try_parse_float(v) is not None: fs[i]+=1
        if dc is None:
            b=0
            for i in range(n):
                if ds[i]>b and hl[i] not in OHLV_ALIASES and hl[i] not in SKIP_ALIASES: b=ds[i]; dc=i
        if pc is None:
            b=0
            for i in range(n):
                if i==dc or hl[i] in SKIP_ALIASES or hl[i] in DATE_ALIASES: continue
                if fs[i]>b: b=fs[i]; pc=i
    if pc is None and dc is not None:
        for i in range(n):
            if i!=dc: pc=i; break
    if dc is None and pc is not None:
        for i in range(n):
            if i!=pc: dc=i; break
    if pc is None and dc is None and n>=2: dc=0; pc=n-1
    if pc is not None and dc is not None and sample_rows:
        tp=td=0
        for row in sample_rows[:10]:
            try:
                if try_parse_float(row[pc]) is not None: tp+=1
                if try_parse_date(row[dc]) is not None: td+=1
            except: pass
        if tp<3 and td<3: pc,dc=dc,pc
        elif tp<3:
            for i in range(n):
                if i==dc: continue
                cnt=sum(1 for r in sample_rows[:10] if len(r)>i and try_parse_float(r[i]) is not None)
                if cnt>=5: pc=i; break
    return pc, dc, sym

def parse_csv_robust(content, filename="unknown.csv"):
    result = {'prices':[],'dates':[],'symbol':os.path.splitext(os.path.basename(filename))[0].upper(),'n_raw_rows':0,'parse_info':{},'errors':[],'warnings':[]}
    content = content.replace('\ufeff','').replace('\r\n','\n').replace('\r','\n')
    lines = [l for l in content.split('\n') if l.strip()]
    if len(lines)<2: result['errors'].append('< 2 lines'); return result
    delim = detect_delimiter(content); result['parse_info']['delimiter']=repr(delim)
    try: all_rows = list(csv.reader(io.StringIO(content), delimiter=delim))
    except Exception as e: result['errors'].append(f'CSV error: {e}'); return result
    all_rows = [r for r in all_rows if any(c.strip() for c in r)]
    if len(all_rows)<2: result['errors'].append('No data'); return result
    fn = sum(1 for c in all_rows[0] if try_parse_float(c.strip()) is not None)
    sn = sum(1 for c in all_rows[1] if try_parse_float(c.strip()) is not None) if len(all_rows)>1 else 0
    hh = fn < sn or fn == 0
    header = all_rows[0] if hh else [f'col{i}' for i in range(len(all_rows[0]))]
    data_rows = all_rows[1:] if hh else all_rows
    result['n_raw_rows']=len(data_rows); result['parse_info']['n_columns']=len(header); result['parse_info']['header']=[h.strip() for h in header]
    pc,dc,ds = identify_columns(header, data_rows[:30])
    if ds!="DATA": result['symbol']=ds
    result['parse_info']['price_col']=pc; result['parse_info']['date_col']=dc
    result['parse_info']['price_col_name']=header[pc].strip() if pc is not None and pc<len(header) else None
    result['parse_info']['date_col_name']=header[dc].strip() if dc is not None and dc<len(header) else None
    if pc is None: result['errors'].append('No price column'); return result
    parsed=[]; errs=0; rc=0
    for row in data_rows:
        rc+=1
        try: ps = row[pc].strip() if pc<len(row) else ''
        except: errs+=1; continue
        p = try_parse_float(ps)
        if p is None or p<=0: errs+=1; continue
        dt=None
        if dc is not None:
            try: dt=try_parse_date(row[dc].strip() if dc<len(row) else '')
            except: pass
        if dt is None: dt=datetime(2000,1,1)+timedelta(minutes=rc)
        parsed.append((dt,p))
    if errs>0: result['warnings'].append(f'{errs}/{rc} rows failed ({errs/rc*100:.1f}%)')
    if not parsed: result['errors'].append('No valid prices'); return result
    parsed.sort(key=lambda x:x[0])
    result['dates']=[d[0] for d in parsed]; result['prices']=[d[1] for d in parsed]
    result['parse_info']['n_parsed']=len(parsed)
    result['parse_info']['price_range']=f"${min(result['prices']):.2f} - ${max(result['prices']):.2f}"
    result['parse_info']['date_range']=f"{result['dates'][0].strftime('%Y-%m-%d')} to {result['dates'][-1].strftime('%Y-%m-%d')}"
    return result

DEFAULT_THRESHOLDS = [0.005,0.0075,0.01,0.015,0.02,0.025,0.03,0.04,0.05]
SYNTHETIC_RUNS = 300
MAX_STASIS_LENGTH = 30

def convert_to_binary(prices, threshold):
    if len(prices)<2: return []
    ref=prices[0]; bw=threshold*ref; upper=ref+bw; lower=ref-bw; bits=[]
    for price in prices:
        if lower<price<upper: continue
        if bw<=0: continue
        x=int((price-ref)/bw)
        if x>0: bits.extend([1]*x)
        elif x<0: bits.extend([0]*abs(x))
        if x!=0: ref=price; bw=threshold*ref; upper=ref+bw; lower=ref-bw
    return bits

def merge_bitstreams(prices, thresholds):
    ab=[]; pt=[]
    for t in thresholds:
        b=convert_to_binary(prices,t); pt.append({'threshold_pct':round(t*100,4),'n_bits':len(b)}); ab.extend(b)
    return ab, pt

def count_stasis_patterns(seq):
    c=defaultdict(int)
    if len(seq)<2: return dict(c)
    cur=1
    for i in range(1,len(seq)):
        if seq[i]!=seq[i-1]: cur+=1
        else:
            if cur>=2: c[cur]+=1
            cur=1
    if cur>=2: c[cur]+=1
    return dict(c)

def calc_alt_rate(seq):
    if len(seq)<2: return 0.5
    return sum(1 for i in range(1,len(seq)) if seq[i]!=seq[i-1])/(len(seq)-1)

def gen_random_baseline(n, runs=SYNTHETIC_RUNS):
    ac=defaultdict(list)
    for _ in range(runs):
        s=[random.randint(0,1)]
        for _ in range(n-1): s.append(1-s[-1] if random.random()<0.5 else s[-1])
        c=count_stasis_patterns(s)
        for l in range(2,MAX_STASIS_LENGTH+1): ac[l].append(c.get(l,0))
    return {'mean':{l:float(np.mean(v)) for l,v in ac.items()}, 'std':{l:float(np.std(v)) for l,v in ac.items()}}

def binom_test(rate, n):
    if n<2: return 0.0, 1.0
    s=int(rate*n); std=np.sqrt(n*0.25); z=(s-0.5*n)/std if std>0 else 0
    try: p=stats.binomtest(s,n,0.5,'two-sided').pvalue
    except: p=stats.binom_test(s,n,0.5,'two-sided')
    return float(z),float(p)

def run_unified_analysis(prices, thresholds=None):
    if thresholds is None: thresholds=DEFAULT_THRESHOLDS
    ub, pts = merge_bitstreams(prices, thresholds)
    if len(ub)<20: return {'error':f'Too short ({len(ub)} bits)','n_prices':len(prices)}
    rc=count_stasis_patterns(ub); ar=calc_alt_rate(ub); zs,pv=binom_test(ar,len(ub)-1)
    bl=gen_random_baseline(len(ub),SYNTHETIC_RUNS)
    mo=max(rc.keys()) if rc else 2; dm=mo+1
    rd={}; rnd={}; ex={}; pd={}; al={}; db={}; dp={}
    for l in range(2,dm+1):
        r=rc.get(l,0); m=bl['mean'].get(l,0); s=bl['std'].get(l,0)
        rd[l]=r; rnd[l]=round(m,2)
        if m>0.01: ex[l]=round(r-m,2); pd[l]=round(((r-m)/m)*100,2); al[l]=round((r-m)/s,2) if s>0 else 0
        else: ex[l]=r if r>0 else 0; pd[l]=None; al[l]=None
    for l in range(3,dm+1):
        prev=rd.get(l-1,0); db[l]=round(prev*0.5,2)
        dp[l]=round((rd.get(l,0)/prev)*100,2) if prev>0 else 0.0
    tr=sum(rd.values()); trn=sum(v for v in rnd.values() if v)
    op=round(((tr-trn)/trn)*100,2) if trn>0 else 0
    dev=(ar-0.5)*100
    if pv<0.01: vd='STRONGLY_NOT_RANDOM'
    elif pv<0.05: vd='NOT_RANDOM'
    elif abs(dev)>0.3: vd='WEAKLY_NOT_RANDOM'
    else: vd='NEAR_RANDOM'
    di='STASIS_BIAS' if ar>0.502 else 'TREND_BIAS' if ar<0.498 else 'NEUTRAL'
    return {'n_prices':len(prices),'thresholds_used':[round(t*100,4) for t in thresholds],'per_threshold':pts,
        'unified_bits':len(ub),'alternation_rate':round(ar,6),'deviation_pct':round(dev,4),'z_score':round(zs,4),'p_value':round(pv,8),
        'real_distribution':rd,'random_distribution':rnd,'excess_real_vs_random':ex,'pct_diff_real_vs_random':pd,
        'alpha_per_length':al,'decay_baseline':db,'decay_pct':dp,'total_real_patterns':tr,'total_random_patterns':round(trn,1),
        'overall_pct_diff':op,'max_observed_length':mo,'display_max_length':dm,'verdict':vd,'direction':di,'is_significant':pv<0.05}

def run_multi_from_session(sid, thresholds=None):
    if thresholds is None: thresholds=DEFAULT_THRESHOLDS
    sess=SESSIONS.get(sid)
    if not sess: return {'error':'Session not found'}
    fr=sess['files']; ar=[]; pr=sess.get('parse_reports',[]); errs=[]
    for idx,f in enumerate(fr):
        if (idx+1)%20==0: print(f"    [{idx+1}/{len(fr)}]")
        try:
            r=run_unified_analysis(f['prices'],thresholds); r['symbol']=f['symbol']; r['filename']=f['name']; r['date_range']=f.get('date_range','?'); ar.append(r)
        except Exception as e: errs.append(f"{f['name']}: {e}")
    agr=defaultdict(int); agn=defaultdict(float); tb=0; rates=[]; gmo=2
    for r in ar:
        if 'error' in r: continue
        tb+=r.get('unified_bits',0); rates.append(r.get('alternation_rate',0.5))
        mo=r.get('max_observed_length',2)
        if mo>gmo: gmo=mo
        for l,v in r.get('real_distribution',{}).items(): agr[int(l)]+=v
        for l,v in r.get('random_distribution',{}).items(): agn[int(l)]+=v
    dm=gmo+1
    for l in range(2,dm+1):
        if l not in agr: agr[l]=0
        if l not in agn: agn[l]=0.0
    if tb>100 and ar: abl=gen_random_baseline(tb//max(len(ar),1),min(SYNTHETIC_RUNS,100))
    else: abl={'std':{}}
    ae={}; ap={}; aa={}; adb={}; adp={}
    for l in range(2,dm+1):
        rv=agr[l]; rn=agn[l]
        if rn>0.01: ae[l]=round(rv-rn,2); ap[l]=round(((rv-rn)/rn)*100,2)
        else: ae[l]=rv if rv>0 else 0; ap[l]=None
        s=abl.get('std',{}).get(l,0)
        if s>0 and ar: aa[l]=round((rv-rn)/(s*np.sqrt(len(ar))),2) if rn>0 else None
        else: aa[l]=None
    for l in range(3,dm+1):
        prev=agr.get(l-1,0); adb[l]=round(prev*0.5,2)
        adp[l]=round((agr.get(l,0)/prev)*100,2) if prev>0 else 0.0
    tar=sum(agr.values()); tan=sum(v for v in agn.values() if v)
    op=round(((tar-tan)/tan)*100,2) if tan>0 else 0
    gr=float(np.mean(rates)) if rates else 0.5; gd=(gr-0.5)*100
    _,gp=binom_test(gr,tb-1) if tb>100 else (0,1.0)
    if gp<0.01: gv='STRONGLY_NOT_RANDOM'
    elif gp<0.05: gv='NOT_RANDOM'
    elif abs(gd)>0.3: gv='WEAKLY_NOT_RANDOM'
    else: gv='NEAR_RANDOM'
    ns=sum(1 for r in ar if r.get('is_significant'))
    return {'files_analyzed':len(ar),'files_attempted':len(fr),'files_significant':ns,'errors':errs,'parse_reports':pr,'individual_results':ar,
        'aggregate':{'total_bits':tb,'n_files':len(ar),'grand_alternation_rate':round(gr,6),'grand_deviation_pct':round(gd,4),
            'grand_p_value':round(gp,8),'grand_verdict':gv,'grand_direction':'STASIS_BIAS' if gd>0.3 else 'TREND_BIAS' if gd<-0.3 else 'NEUTRAL',
            'real_distribution':dict(agr),'random_distribution':{l:round(v,2) for l,v in agn.items()},
            'excess_real_vs_random':ae,'pct_diff_real_vs_random':ap,'alpha_per_length':aa,
            'decay_baseline':adb,'decay_pct':adp,'overall_pct_diff':op,'display_max_length':dm}}

@app.route('/')
def index(): return render_template('proof.html')

@app.route('/health')
def health():
    cleanup_old_sessions()
    return jsonify({'status':'ok','version':'6.1','sessions':len(SESSIONS)})

@app.route('/session/create', methods=['POST'])
def session_create():
    cleanup_old_sessions(); sid=str(uuid.uuid4()); SESSIONS[sid]={'files':[],'parse_reports':[],'created':time.time()}
    return jsonify({'session_id':sid})

@app.route('/session/upload', methods=['POST'])
def session_upload():
    sid=request.form.get('session_id','')
    if sid not in SESSIONS: return jsonify({'error':'Invalid session'}),400
    sess=SESSIONS[sid]; files=request.files.getlist('file')
    if not files: return jsonify({'error':'No file'}),400
    f=files[0]
    if not f.filename: return jsonify({'error':'Empty'}),400
    try:
        raw=f.read(); content=None
        for enc in ['utf-8-sig','utf-8','latin-1','cp1252','ascii']:
            try: content=raw.decode(enc); break
            except: continue
        if not content: content=raw.decode('utf-8',errors='replace')
        del raw
        parsed=parse_csv_robust(content,f.filename); del content
        rpt={'filename':f.filename,'symbol':parsed['symbol'],'n_parsed':parsed['parse_info'].get('n_parsed',0),
            'price_col':parsed['parse_info'].get('price_col_name','?'),'price_range':parsed['parse_info'].get('price_range','?'),
            'date_range':parsed['parse_info'].get('date_range','?'),'warnings':parsed['warnings'],'errors':parsed['errors']}
        sess['parse_reports'].append(rpt)
        if parsed['errors'] or len(parsed['prices'])<20:
            return jsonify({'status':'skipped','filename':f.filename,'reason':parsed['errors'][0] if parsed['errors'] else f"Only {len(parsed['prices'])} prices",'total':len(sess['files'])})
        sess['files'].append({'name':f.filename,'symbol':parsed['symbol'],'prices':parsed['prices'],'date_range':parsed['parse_info'].get('date_range','?')})
        return jsonify({'status':'ok','filename':f.filename,'symbol':parsed['symbol'],'n_prices':len(parsed['prices']),'total':len(sess['files'])})
    except Exception as e: return jsonify({'status':'error','filename':f.filename,'error':str(e)}),200

@app.route('/session/run', methods=['POST'])
def session_run():
    data=request.get_json() or {}; sid=data.get('session_id','')
    if sid not in SESSIONS: return jsonify({'error':'Invalid session'}),400
    sess=SESSIONS[sid]
    if not sess['files']: return jsonify({'error':'No files'}),400
    th=DEFAULT_THRESHOLDS; ts=data.get('thresholds','')
    if ts:
        try: th=[float(x.strip()) for x in ts.split(',') if x.strip()]
        except: pass
    print(f"\n  Analyzing {len(sess['files'])} files (session {sid[:8]})")
    result=run_multi_from_session(sid,th)
    del SESSIONS[sid]; gc.collect()
    print(f"  Done: {result.get('files_analyzed',0)} analyzed")
    return jsonify(result)

@app.route('/analyze-demo')
def analyze_demo():
    np.random.seed(42); n=10000; rets=np.random.normal(0,0.01,n); prices=[100.0]
    for r in rets: prices.append(prices[-1]*(1+r))
    result=run_unified_analysis(prices,DEFAULT_THRESHOLDS)
    result['symbol']='RANDOM_WALK'; result['filename']='synthetic_demo.csv'; result['date_range']='2020-01-01 to 2024-12-31'
    return jsonify({'files_analyzed':1,'files_attempted':1,'files_significant':1 if result.get('is_significant') else 0,'errors':[],
        'parse_reports':[{'filename':'synthetic_demo.csv','symbol':'RANDOM_WALK','n_parsed':n,'price_col':'synthetic',
            'price_range':f"${min(prices):.2f}-${max(prices):.2f}",'date_range':'2020-01-01 to 2024-12-31','warnings':['Synthetic'],'errors':[]}],
        'individual_results':[result],
        'aggregate':{'total_bits':result.get('unified_bits',0),'n_files':1,'grand_alternation_rate':result.get('alternation_rate',0.5),
            'grand_deviation_pct':result.get('deviation_pct',0),'grand_p_value':result.get('p_value',1),'grand_verdict':result.get('verdict','NEAR_RANDOM'),
            'grand_direction':result.get('direction','NEUTRAL'),'real_distribution':result.get('real_distribution',{}),
            'random_distribution':result.get('random_distribution',{}),'excess_real_vs_random':result.get('excess_real_vs_random',{}),
            'pct_diff_real_vs_random':result.get('pct_diff_real_vs_random',{}),'alpha_per_length':result.get('alpha_per_length',{}),
            'decay_baseline':result.get('decay_baseline',{}),'decay_pct':result.get('decay_pct',{}),
            'overall_pct_diff':result.get('overall_pct_diff',0),'display_max_length':result.get('display_max_length',10)},
        'is_demo':True})

@app.errorhandler(413)
def too_large(e): return jsonify({'error':'File too large (max 20MB)'}),413

if __name__ == '__main__':
    print("="*60); print("  PROOF ENGINE v6.1 — IMPROVED CHART SPACING"); print("="*60)
    print(f"\n  http://127.0.0.1:5000\n")
    app.run(debug=True,host='0.0.0.0',port=5000,threaded=True)