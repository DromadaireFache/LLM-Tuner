import argparse
import datetime
import fcntl
import json
import math
import os
import pathlib
import re
import signal
import shlex
import subprocess
import sys
import time

from . import __version__
from .model import GGUF, duration, normalized, read_presets, supported_flags
from .reporting import write_report
from .runtime import Runtime, Service, hardware, http, ContextSizeError
from .search import Search, finalists, key


def positive_count(value):
    try:n=int(value)
    except ValueError:raise argparse.ArgumentTypeError('contexts must be a single integer, minimum 1')
    if n<1:raise argparse.ArgumentTypeError('contexts must be at least 1')
    return n


def sample_contexts(count,capacity,decode):
    """Even spacing over usable input length, rounded for readability without duplicates."""
    low=1024;high=capacity-decode
    if count<1:raise ValueError('contexts must be at least 1')
    if high<low:raise ValueError('ctx-size must fit 1K input plus decode tokens; use --context-lengths for smaller tests')
    if count>high-low+1:raise ValueError('Too many contexts to choose distinct input lengths within this capacity')
    step=1024 if capacity>=8192 else 256 if high-low>=256 else 1
    if count==1:return [max(low,min(high,round(((low+high)/2)/step)*step))]
    points=[low]+[max(low,min(high,round((low+i*(high-low)/(count-1))/step)*step)) for i in range(1,count-1)]+[high]
    if len(set(points))!=count:points=[low+round(i*(high-low)/(count-1)) for i in range(count)]
    return points


def parser():
    p=argparse.ArgumentParser(description='Automatically benchmark and tune local llama.cpp models within a time budget. Reports recommendations; never edits models.ini.',epilog='Examples: llm-tune inkling2-mini --budget 10min --contexts 1 | llm-tune qwen3.6-35b-quick --budget 1hr --contexts 3 | llm-tune ./new.gguf --budget 8hr --ctx-size 65536 --contexts 5')
    p.add_argument('model',nargs='?',help='Preset name or GGUF path; omit for an interactive menu; use recover with --run-dir to recover a crashed run')
    p.add_argument('--budget',default='10min',help='Maximum tuning time: 10min, 1hr, 8hr, or 1h30m (default: 10min)')
    group=p.add_mutually_exclusive_group()
    group.add_argument('--contexts',type=positive_count,help='Number of evenly spaced input lengths from 1K to ctx-size minus decode tokens; 1 selects the midpoint (default: 3)')
    group.add_argument('--context-lengths',help='Advanced: exact input token counts, e.g. 512,2048,8192; replaces automatic sampling')
    p.add_argument('--ctx-size',type=int,help='Fixed context capacity (default: preset capacity; bare GGUF: 8192, or enough for explicit context-lengths)')
    p.add_argument('--capacities',help='Optional comma-separated context capacities to search; each profile must fit its candidate capacity')
    p.add_argument('--decode',type=int,default=64,help='Output tokens actually measured per profile (default: 64)')
    p.add_argument('--output-lengths',default='32,128,512',help='Output lengths for projected combined-time rankings')
    p.add_argument('--screen-tokens',type=int,default=256,help='Short input used to screen candidates (default: 256)')
    p.add_argument('--screen-decode',type=int,default=24)
    p.add_argument('--search-fraction',type=float,default=.4,help='Budget fraction for screening; remainder profiles finalists (default: 0.4)')
    p.add_argument('--finalists',type=int,default=3)
    p.add_argument('--repeats',type=int,default=1,help='Profile repetitions per input length; report uses medians')
    p.add_argument('--max-trials',type=int,help='Optional cap on screened candidates')
    p.add_argument('--models-ini',default='~/models/models.ini')
    p.add_argument('--server-bin',default='~/llama.cpp/build/bin/llama-server')
    p.add_argument('--service',default='llmserver',help='User systemd service to pause when idle and restore afterward; none leaves services alone')
    p.add_argument('--router-url',default='http://127.0.0.1:8080')
    p.add_argument('--ignore-router-overrides',action='store_true',help='Explore settings the parent CLI would override; recommendations may then require parent startup changes')
    p.add_argument('--prompt-file',help='Representative text; repeat token IDs if shorter than the longest requested input')
    p.add_argument('--check-prompt',default='What is 17 times 23? Give the numerical result in your final answer.')
    p.add_argument('--check-tokens',type=int,default=512,help='Maximum generation tokens for the normal-stopping chat smoke check (default: 512)')
    p.add_argument('--expect',default='391',help='Expected substring for bounded chat smoke checks; a miss is reported as a quality warning')
    p.add_argument('--vram-headroom',type=float,default=256,help='Minimum measured free VRAM in MiB')
    p.add_argument('--ram-headroom',type=float,default=384,help='Minimum system available RAM in MiB')
    p.add_argument('--max-swap-out',type=float,default=128,help='Maximum swap-out during a request/startup in MiB')
    p.add_argument('--output-dir',default='~/llm-tuning-runs')
    p.add_argument('--summary-dir',default=str(pathlib.Path(__file__).resolve().parents[1]),help='Directory for latest <model>.md summaries (default: llm-tuner installation directory)')
    p.add_argument('--run-dir',help='Existing run directory for recover')
    p.add_argument('--list',action='store_true',help='List available model presets and exit')
    p.add_argument('--plan',action='store_true',help='Inspect model and show constraints without stopping or benchmarking anything')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--version',action='version',version=__version__)
    return p


def csv_ints(value):
    try:out=sorted(set(int(v.strip()) for v in value.split(',')))
    except ValueError:raise ValueError('Expected comma-separated positive integers')
    if not out or min(out)<=0:raise ValueError('Lengths must be positive')
    return out


def make_text(max_tokens):
    # Diverse, deterministic prose and numeric fields exercise more experts than one repeated sentence.
    parts=['Read the following operations records and continue their format.\n']
    for i in range(max(100,max_tokens//12+100)):
        parts.append(f'Record {i}: warehouse {(i*7)%19} processed {31+i%89} parcels. The sensor reading was {18+i%13} degrees; maintenance inspected conveyor {i%8}. Delivery reports cover inventory, routing, equipment reliability, and staffing.\n')
    return ''.join(parts)


def recover(args):
    if not args.run_dir:raise ValueError('recover requires --run-dir /path/to/run')
    lockpath=pathlib.Path('~/.cache/llm-tuner.lock').expanduser();lockpath.parent.mkdir(parents=True,exist_ok=True)
    recovery_lock=open(lockpath,'a')
    try:fcntl.flock(recovery_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise RuntimeError('A tuner is still running; interrupt it normally instead of recovering it')
    root=pathlib.Path(args.run_dir).expanduser().resolve();state=json.loads((root/'state.json').read_text())
    pid=state.get('trial_pid');alias=state.get('trial_alias','')
    if pid and alias.startswith('llm-tuner-'):
        try:
            argv=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            if alias.encode() not in argv:raise RuntimeError('PID no longer belongs to this tuning run; refusing to kill it')
            os.killpg(pid,signal.SIGTERM)
            for _ in range(20):
                if not pathlib.Path(f'/proc/{pid}').exists():break
                time.sleep(.05)
            if pathlib.Path(f'/proc/{pid}').exists() and alias.encode() in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0'):os.killpg(pid,signal.SIGKILL)
        except ProcessLookupError:pass
        except FileNotFoundError:pass
    Service(state.get('service_name','none'),state.get('router_url',args.router_url),state,root).restore()
    state.pop('trial_pid',None);state['status']='recovered';(root/'state.json').write_text(json.dumps(state,indent=2))
    print('Recovered service state. '+str(root/'state.json'))
    recovery_lock.close()
    return 0


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    try:
        if args.model=='recover':return recover(args)
        budget=duration(args.budget)
        if budget<2:raise ValueError('Budget must be at least two seconds')
        contexts=csv_ints(args.context_lengths) if args.context_lengths else None;outputs=csv_ints(args.output_lengths)
        if args.decode<2 or args.screen_decode<2 or args.screen_tokens<8 or args.repeats<1 or args.finalists<1 or args.check_tokens<1:raise ValueError('Use decode >= 2, screen-tokens >= 8, repeats/finalists/check-tokens >= 1')
        if args.max_trials is not None and args.max_trials<1:raise ValueError('max-trials must be at least one')
        if min(args.ram_headroom,args.vram_headroom,args.max_swap_out)<0:raise ValueError('Resource limits must be non-negative')
        if not 0<args.search_fraction<.9:raise ValueError('search-fraction must be between 0 and 0.9')
        presets=read_presets(args.models_ini) if pathlib.Path(args.models_ini).expanduser().exists() else {}
        if args.list:
            for name,c in presets.items():print(f'{name:25} {c.get("model","(no local GGUF path)")}')
            return 0
        if not args.model:
            if not sys.stdin.isatty():p.error('Specify a preset or GGUF path, or use --list')
            names=list(presets)
            for i,name in enumerate(names,1):print(f'{i:2}. {name}')
            choice=input('Model number, preset name, or GGUF path: ').strip()
            args.model=names[int(choice)-1] if choice.isdigit() and 1<=int(choice)<=len(names) else choice
            chosen=input(f'Time budget [{args.budget}]: ').strip()
            if chosen:budget=duration(chosen)
            if budget<2:raise ValueError('Budget must be at least two seconds')
        base=dict(presets.get(args.model,{}));model_path=base.get('model',args.model)
        if not pathlib.Path(model_path).expanduser().is_file():raise ValueError('Model must resolve to a local GGUF file: '+str(model_path))
        model=GGUF(model_path);base['model']=model.path
        binary=pathlib.Path(args.server_bin).expanduser().resolve()
        if not binary.is_file():raise ValueError('llama-server not found: '+str(binary))
        help_text=subprocess.check_output([str(binary),'--help'],text=True,stderr=subprocess.STDOUT,timeout=15)
        supported=supported_flags(help_text)
        required={'model','ctx-size','batch-size','ubatch-size','host','port','alias'}
        if required-supported:raise ValueError('llama-server lacks required flags: '+str(required-supported))
        version=subprocess.check_output([str(binary),'--version'],text=True,stderr=subprocess.STDOUT,timeout=15).strip()
        state={'status':'preparing','router_url':args.router_url}
        # Read effective parent capacity before selecting automatic input lengths.
        service=Service(args.service,args.router_url,state,pathlib.Path('/nonexistent'))
        forced=service.inspect(allow_busy=args.plan)
        if args.ignore_router_overrides:forced={}
        cap=args.ctx_size or int(forced.get('ctx-size',0)) or int(base.get('ctx-size',0)) or (math.ceil((max(contexts)+args.decode)/1024)*1024 if contexts else 8192)
        capacities=csv_ints(args.capacities) if args.capacities else [cap]
        if args.ctx_size and args.capacities:raise ValueError('Choose --ctx-size or --capacities, not both')
        if 'ctx-size' in forced and any(int(forced['ctx-size'])!=n for n in capacities):raise ValueError('Router CLI pins ctx-size; use matching capacities or --ignore-router-overrides')
        if contexts is None:contexts=sample_contexts(args.contexts or 3,max(capacities),args.decode)
        if max(capacities)<max(contexts)+args.decode:raise ValueError('Largest capacity must fit largest input plus measured decode tokens')
        if min(capacities)<args.screen_decode+8:raise ValueError('Each capacity must fit at least eight screening tokens plus screen-decode tokens')
        args.screen_tokens=min(args.screen_tokens,min(capacities)-args.screen_decode)
        base.update({'ctx-size':capacities[-1],'batch-size':int(base.get('batch-size',512)),'ubatch-size':int(base.get('ubatch-size',128)),'threads':int(base.get('threads',os.cpu_count() or 1)),'parallel':1,'cache-ram':256,'ctx-checkpoints':1})
        base.setdefault('jinja',True);base.setdefault('flash-attn','on');base.setdefault('load-mode','mmap')
        base=normalized(base,forced)
        if args.plan:
            print(json.dumps({'model':model.summary(),'base':base,'context_count':len(contexts),'contexts':contexts,'capacities':capacities,'budget_seconds':budget,'router_constraints':forced,'service_would_pause':state.get('service_was_active',False),'hardware':hardware()},indent=2));return 0
        run_name=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+re.sub(r'[^a-zA-Z0-9_.-]+','-',pathlib.Path(args.model).stem)[:60]+f'-{os.getpid()}'
        root=pathlib.Path(args.output_dir).expanduser().resolve()/run_name;root.mkdir(parents=True)
        lockpath=pathlib.Path('~/.cache/llm-tuner.lock').expanduser();lockpath.parent.mkdir(parents=True,exist_ok=True)
        lock=open(lockpath,'w')
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another tuner is running; use one tuning process at a time')
        service.root=root;service.persist()
        start=time.monotonic();deadline=start+budget
        runtime=Runtime(binary,root,supported,deadline-min(2,budget*.1),{'ram_floor':args.ram_headroom,'vram_floor':args.vram_headroom,'swap_limit':args.max_swap_out},state)
        report_name=args.model if args.model in presets else pathlib.Path(model.path).stem
        meta={'version':__version__,'model':model.summary(),'model_name':re.sub(r'[^a-zA-Z0-9_.-]+','-',report_name),'summary_directory':str(pathlib.Path(args.summary_dir).expanduser().resolve()),'budget_seconds':budget,'contexts':contexts,'context_count':len(contexts),'context_sampling':'explicit' if args.context_lengths else 'evenly spaced from 1K to capacity minus decode','capacities':capacities,'decode_tokens':args.decode,'output_lengths':outputs,'router_constraints':forced,'server_version':version,'recovery_command':f'llm-tune recover --run-dir {shlex.quote(str(root))}','started_at':datetime.datetime.now().astimezone().isoformat()}
        results=[];configs={};ids={};tokens=None;status='running';trial_counter=0
        def save():
            meta['elapsed_seconds']=time.monotonic()-start
            write_report(root,meta,results,configs,status)
        def cid(c):
            k=key(c)
            if k not in ids:
                name=f'C{len(ids)+1:03}';ids[k]=name;configs[name]=c
            return ids[k]
        def begin(c):
            nonlocal trial_counter,tokens
            trial_counter+=1;identifier=f'{trial_counter:04}-{cid(c)}'
            loaded=runtime.start(c,identifier)
            if loaded['status']=='invalid_context':
                meta['context_mismatch']={k:loaded[k] for k in ('requested_context','actual_context')}
                record(c,loaded,'startup')
                raise ContextSizeError(loaded['error'])
            if loaded['status']!='ready':return loaded
            if tokens is None:
                text=pathlib.Path(args.prompt_file).expanduser().read_text() if args.prompt_file else make_text(max(contexts))
                raw=http(runtime.base,'/tokenize',{'content':text,'add_special':True},timeout=max(.1,runtime.remaining()))['tokens']
                if not raw:raise RuntimeError('Tokenizer returned no tokens')
                needed=max(max(contexts),args.screen_tokens)
                tokens=(raw*((needed+len(raw)-1)//len(raw)))[:needed]
                (root/'prompt-tokens.json').write_text(json.dumps(tokens))
                meta['prompt_source']=str(pathlib.Path(args.prompt_file).expanduser()) if args.prompt_file else 'deterministic varied operations log'
                meta['prompt_repeated']=len(raw)<needed
            if runtime.remaining()>2:
                warm=runtime.completion(tokens[:min(64,len(tokens))],4,'warmup')
                if warm['status']!='ok':return warm
            return loaded
        def record(c,r,stage):
            r.update(config_id=cid(c),config=c,stage=stage);results.append(r);save()
            if r['status']=='ok':print(f'{stage:7} {cid(c)} input={r["input_tokens"]:6} prefill={r["timings"]["prompt_per_second"]:8.2f} t/s decode={r["timings"]["predicted_per_second"]:7.2f} t/s | remaining {runtime.remaining():.0f}s',flush=True)
            else:print(f'{stage:7} {cid(c)} {r["status"]}: {str(r.get("error",""))[-200:]}',flush=True)
        oldterm=signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            print(f'Run: {root}\nInput lengths: {contexts}; capacities: {capacities}; measured output: {args.decode} tokens\nBudget: {budget:.0f}s. Temporary servers use localhost. Original service/models will be restored. No presets will be edited.',flush=True)
            service.pause();meta['hardware']=hardware();search=Search(model,base,capacities,meta['hardware'],supported,forced,args.seed)
            search_end=start+budget*args.search_fraction
            screened=0
            while runtime.remaining()>2 and time.monotonic()<search_end:
                if args.max_trials is not None and screened>=args.max_trials:break
                c,family=search.next()
                if c is None:break
                print(f'Trying {cid(c)} ({family})',flush=True)
                overall_deadline=runtime.deadline
                runtime.deadline=min(overall_deadline,search_end)
                try:
                    r=begin(c)
                    if r['status']=='ready':
                        load_s=r.get('load_s')
                        r=runtime.completion(tokens[:args.screen_tokens],args.screen_decode,'screen');r['load_s']=load_s
                finally:runtime.deadline=overall_deadline
                record(c,r,'screen');search.feedback(c,r);screened+=1;runtime.stop()
            chosen=finalists(results,args.finalists)
            for c in chosen:
                if runtime.remaining()<3:break
                r=begin(c)
                if r['status']!='ready':record(c,r,'profile');runtime.stop();continue
                load_s=r.get('load_s');measured=False
                for n in contexts:
                    if n+args.decode>int(c['ctx-size']):
                        record(c,{'status':'capacity_too_small','error':'Input + output exceeds this candidate capacity','input_tokens':n},'profile');continue
                    # A conservative admission check avoids spending the entire remaining budget on a known-too-long request.
                    prior=[t for t in results if t.get('config_id')==cid(c) and t['status']=='ok']
                    rate=prior[-1]['timings']['prompt_per_second'] if prior else 1
                    dec=prior[-1]['timings']['predicted_per_second'] if prior else 1
                    estimate=n/rate+args.decode/dec
                    if estimate*1.15>runtime.remaining():
                        record(c,{'status':'skipped_budget','error':f'Estimated {estimate:.1f}s exceeds remaining budget','input_tokens':n},'profile');continue
                    for repeat in range(args.repeats):
                        if repeat and estimate*1.15>runtime.remaining():break
                        r=runtime.completion(tokens[:n],args.decode,'profile');r['load_s']=load_s
                        record(c,r,'profile');measured=True
                        if r['status']!='ok':break
                    if r['status'] not in ('ok','skipped_budget','capacity_too_small'):break
                if measured and runtime.remaining()>3:
                    check=runtime.chat_check(args.check_prompt,args.expect,args.check_tokens)
                    results.append({'config_id':cid(c),'config':c,'stage':'chat_check','status':check['status'],'check':check});save()
                runtime.stop()
            # Use spare time for independent repeats, without inventing long-context coverage.
            profile_ok=[t for t in results if t.get('stage')=='profile' and t['status']=='ok']
            if profile_ok and runtime.remaining()>10:
                best=min(profile_ok,key=lambda t:t['input_tokens']/t['timings']['prompt_per_second']+128/t['timings']['predicted_per_second'])
                c=best['config'];n=best['input_tokens'];estimate=best['wall_s']
                if estimate*1.3+best.get('load_s',0)<runtime.remaining():
                    r=begin(c)
                    if r['status']=='ready':record(c,runtime.completion(tokens[:n],args.decode,'repeat'),'repeat')
                    runtime.stop()
            status='complete' if any(t.get('stage')=='profile' and t['status']=='ok' for t in results) else 'no_profile_completed' if chosen else 'no_successful_configuration'
        except ContextSizeError as e:
            status='invalid_context';meta['error']=str(e)
            print('Context configuration error: '+str(e),file=sys.stderr,flush=True)
        except KeyboardInterrupt:status='interrupted';print('Stopping tests and restoring service...',flush=True)
        except Exception as e:
            status='error';meta['error']=repr(e);print('Tuning error: '+repr(e),file=sys.stderr,flush=True)
        finally:
            runtime.stop();service.restore();signal.signal(signal.SIGTERM,oldterm)
            state['status']=status;service.persist();meta['service_restore']=state;save();lock.close()
        print(f'Finished: {status}\nSummary: {meta.get("markdown_summary", "not written")}\nReport: {root}/report.html\nText: {root}/report.txt\nData: {root}/results.json\nINI recommendations: {root}/recommendations.ini',flush=True)
        return 0 if status=='complete' else 130 if status=='interrupted' else 2 if status=='invalid_context' else 1
    except (ValueError,RuntimeError,OSError,subprocess.SubprocessError,KeyError) as e:
        print('llm-tune: '+str(e),file=sys.stderr);return 2
