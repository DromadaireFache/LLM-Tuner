import json
import math
import multiprocessing as mp
import os
import pathlib
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request

from .model import PARENT_KEYS, cli_args, parse_argv


class ContextSizeError(RuntimeError):
    """A server context mismatch cannot be repaired by another speed candidate."""


def http(base,path,payload=None,timeout=3):
    data=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request(base+path,data,{'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=max(.05,timeout)) as res:return json.load(res)


def hardware():
    h={'logical_cpus':os.cpu_count() or 1,'physical_cpus':os.cpu_count() or 1}
    cores=set()
    for p in pathlib.Path('/sys/devices/system/cpu').glob('cpu[0-9]*/topology/core_id'):
        try:cores.add((p.with_name('physical_package_id').read_text().strip(),p.read_text().strip()))
        except OSError:pass
    if cores:h['physical_cpus']=len(cores)
    try:
        mem=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
        h['ram_total_mib']=int(mem['MemTotal'].split()[0])/1024
    except OSError:pass
    g=gpu_sample()
    if g:h.update({'gpu_total_mib':g['total'],'gpu_free_mib':g['total']-g['used']})
    return h


def gpu_sample():
    try:
        s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,stderr=subprocess.DEVNULL,timeout=1).splitlines()[0]
        a=list(map(float,s.split(',')));return {'total':a[0],'used':a[1],'util':a[2]}
    except (OSError,ValueError,subprocess.SubprocessError,IndexError):return None


class Sampler:
    def __init__(self,pid,limits):self.pid=pid;self.limits=limits;self.samples=[];self.last=0
    def sample(self,force=False):
        now=time.monotonic()
        if not force and now-self.last<1:return None
        self.last=now;s={'t':now}
        try:
            m=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
            v=dict(line.split() for line in pathlib.Path('/proc/vmstat').read_text().splitlines())
            s.update(available=int(m['MemAvailable'].split()[0])/1024,swap=(int(m['SwapTotal'].split()[0])-int(m['SwapFree'].split()[0]))/1024,swapout=int(v['pswpout']))
            st=pathlib.Path(f'/proc/{self.pid}/stat').read_text().split()
            io=dict(line.split(':',1) for line in pathlib.Path(f'/proc/{self.pid}/io').read_text().splitlines())
            s.update(cpu=int(st[13])+int(st[14]),rss=int(st[23])*os.sysconf('SC_PAGE_SIZE')/1024**2,reads=int(io['read_bytes']),majflt=int(st[11]))
        except (OSError,ValueError,KeyError):pass
        g=gpu_sample()
        if g:s.update(gpu=g['used'],gpu_free=g['total']-g['used'],gpu_util=g['util'])
        self.samples.append(s)
        if 'available' in s and s['available']<self.limits['ram_floor']:return 'RAM headroom below configured floor'
        if 'gpu_free' in s and s['gpu_free']<self.limits['vram_floor']:return 'VRAM headroom below configured floor'
        old=next((x for x in self.samples if 'swapout' in x),None)
        if old and (s.get('swapout',old['swapout'])-old['swapout'])*os.sysconf('SC_PAGE_SIZE')/1024**2>self.limits['swap_limit']:return 'Swap-out exceeded configured limit'
        return None
    def summary(self):
        out={};ss=self.samples
        for key,fn in [('gpu',max),('rss',max),('available',min),('gpu_free',min)]:
            vals=[s[key] for s in ss if key in s]
            if vals:out[key+'_mib']=fn(vals)
        for key in ('gpu_util',):
            vals=[s[key] for s in ss if key in s]
            if vals:out[key+'_mean']=sum(vals)/len(vals)
        usable=[s for s in ss if 'cpu' in s]
        if len(usable)>1:
            a,b=usable[0],usable[-1];dt=b['t']-a['t']
            out.update(cpu_percent=(b['cpu']-a['cpu'])/os.sysconf('SC_CLK_TCK')/dt*100,read_bytes=b['reads']-a['reads'],major_faults=b['majflt']-a['majflt'])
        usable=[s for s in ss if 'swapout' in s]
        if usable:
            a,b=usable[0],usable[-1]
            out.update(swap_start_mib=a['swap'],swap_end_mib=b['swap'],swap_out_mib=(b['swapout']-a['swapout'])*os.sysconf('SC_PAGE_SIZE')/1024**2)
        return out


def _completion_worker(conn,base,payload,timeout):
    start=time.monotonic();first=None;chunks=[];final={}
    try:
        req=urllib.request.Request(base+'/completion',json.dumps(payload).encode(),{'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=max(.1,timeout)) as response:
            for line in response:
                if not line.startswith(b'data: '):continue
                data=line[6:].strip()
                if data==b'[DONE]':break
                d=json.loads(data)
                if d.get('content'):
                    if first is None:first=time.monotonic()-start
                    chunks.append(d['content'])
                if d.get('stop'):final=d
        timings=final.get('timings') or {}
        text=''.join(chunks)
        valid=all(math.isfinite(timings.get(k,0)) and timings.get(k,0)>0 for k in ('prompt_per_second','predicted_per_second'))
        # Obvious single-token corruption is rejected; semantic correctness is checked separately.
        garbage=len(text)>24 and max((text.count(c) for c in set(text)),default=0)/len(text)>.90
        conn.send({'status':'ok' if valid and not garbage else 'invalid_output','timings':timings,'ttft_s':first,'wall_s':time.monotonic()-start,'content':text,'tokens_cached':timings.get('cache_n',0),'error':None if valid and not garbage else 'Missing/non-finite timings or repeated-character output'})
    except Exception as e:conn.send({'status':'request_error','error':repr(e),'wall_s':time.monotonic()-start})
    finally:conn.close()


class Runtime:
    def __init__(self,binary,run_dir,supported,deadline,limits,state):
        self.binary=str(binary);self.root=pathlib.Path(run_dir);self.supported=supported;self.deadline=deadline;self.limits=limits;self.state=state
        self.proc=None;self.log=None;self.sampler=None;self.base=None;self.alias=None
    def remaining(self):return max(0,self.deadline-time.monotonic())
    def save_state(self):
        tmp=self.root/'state.tmp';tmp.write_text(json.dumps(self.state,indent=2));tmp.replace(self.root/'state.json')
    def start(self,config,trial_id):
        self.stop()
        sock=socket.socket();sock.bind(('127.0.0.1',0));port=sock.getsockname()[1];sock.close()
        self.alias='llm-tuner-'+self.root.name+'-'+str(trial_id);self.base=f'http://127.0.0.1:{port}'
        args=[self.binary]+cli_args(config,self.supported)+['--host','127.0.0.1','--port',str(port),'--alias',self.alias]
        if 'fit' in self.supported:args+=['--fit','off']
        if 'log-verbosity' in self.supported:args+=['--log-verbosity','4']
        if 'log-colors' in self.supported:args+=['--log-colors','off']
        (self.root/f'{trial_id}.command.json').write_text(json.dumps(args,indent=2))
        self.log=open(self.root/f'{trial_id}.log','w');start=time.monotonic()
        self.proc=subprocess.Popen(args,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        self.state.update(trial_pid=self.proc.pid,trial_alias=self.alias);self.save_state()
        self.sampler=Sampler(self.proc.pid,self.limits)
        while self.remaining()>0:
            if self.proc.poll() is not None:
                tail=(self.root/f'{trial_id}.log').read_text(errors='replace')[-6000:]
                return {'status':'oom' if any(w in tail.lower() for w in ('out of memory','failed to allocate','bad_alloc')) else 'startup_error','error':tail,'load_s':time.monotonic()-start}
            violation=self.sampler.sample()
            if violation:return {'status':'resource_limit','error':violation,'resources':self.sampler.summary(),'load_s':time.monotonic()-start}
            try:
                if http(self.base,'/health',timeout=min(.25,self.remaining())).get('status')=='ok':
                    props=http(self.base,'/props',timeout=min(1,self.remaining()))
                    if props.get('model_alias') not in (None,self.alias):return {'status':'startup_error','error':'Port collision: health response belongs to another model alias'}
                    if props.get('model_path') and pathlib.Path(props['model_path']).resolve()!=pathlib.Path(config['model']).resolve():return {'status':'startup_error','error':'Effective model path does not match selected GGUF'}
                    actual=props.get('default_generation_settings',{}).get('n_ctx')
                    if actual is not None and int(actual)!=int(config['ctx-size']):
                        requested=int(config['ctx-size']);actual=int(actual)
                        return {'status':'invalid_context','requested_context':requested,'actual_context':actual,'load_s':time.monotonic()-start,'error':f'Requested context capacity {requested:,}, but llama-server exposes a usable request slot of {actual:,} tokens. Check the model training-context cap and parent/per-slot settings. Choose a matching --ctx-size; every input length plus --decode must fit it. Changing batch size, GPU layers, or threads cannot fix this mismatch.'}
                    return {'status':'ready','load_s':time.monotonic()-start,'props':{k:props.get(k) for k in ('model_path','model_alias','build_info')}}
            except (OSError,ValueError,urllib.error.URLError):pass
            time.sleep(min(.1,self.remaining()))
        return {'status':'budget_exhausted','error':'Deadline reached during model load'}
    def completion(self,tokens,predict,stage):
        if self.remaining()<=0:return {'status':'budget_exhausted','error':'Deadline reached'}
        context=mp.get_context('spawn');recv,send=context.Pipe(duplex=False)
        payload={'model':self.alias,'prompt':tokens,'n_predict':predict,'temperature':0,'seed':42,'cache_prompt':False,'ignore_eos':True,'stream':True,'return_progress':True}
        worker=context.Process(target=_completion_worker,args=(send,self.base,payload,self.remaining()),daemon=True);worker.start();send.close()
        # Keep separate per-request metrics: startup/idle data must not contaminate measured CPU/IO.
        request_sampler=Sampler(self.proc.pid,self.limits)
        result=None
        try:
            while self.remaining()>0:
                if recv.poll(.1):
                    try:result=recv.recv()
                    except EOFError:result={'status':'worker_error','error':'HTTP worker exited without a result'}
                    break
                violation=request_sampler.sample()
                if violation:
                    result={'status':'resource_limit','error':violation};break
                if self.proc.poll() is not None:
                    result={'status':'server_exit','error':'Server exited while processing the request'};break
                if not worker.is_alive() and not recv.poll():
                    result={'status':'worker_error','error':'HTTP worker died'};break
            if result is None:result={'status':'budget_exhausted','error':'Request stopped at deadline'}
        finally:
            if worker.is_alive():worker.terminate()
            worker.join(timeout=.5)
            if worker.is_alive():worker.kill();worker.join(timeout=.5)
            recv.close()
        request_sampler.sample(force=True)
        result.update(resources=request_sampler.summary(),stage=stage,input_tokens=len(tokens),output_tokens=predict)
        if result.get('status')=='ok' and (result['tokens_cached'] or result['timings'].get('prompt_n')!=len(tokens)):
            result.update(status='invalid_measurement',error='Prompt cache was used or evaluated token count differed')
        return result
    def chat_check(self,prompt,expected,max_tokens=512):
        if self.remaining()<2:return {'status':'skipped','reason':'Insufficient budget for chat check'}
        # A bounded separate process allows the same hard deadline as throughput requests.
        context=mp.get_context('spawn');recv,send=context.Pipe(duplex=False)
        worker=context.Process(target=_chat_worker,args=(send,self.base,self.alias,prompt,self.remaining(),max_tokens),daemon=True);worker.start();send.close()
        try:
            while self.remaining()>0:
                if recv.poll(.1):
                    try:d=recv.recv()
                    except EOFError:return {'status':'error','error':'Chat worker exited'}
                    text=d.get('content','')+' '+d.get('reasoning_content','')
                    d['expected_found']=expected.lower() in text.lower() if expected else None
                    d['verification']='passed' if d.get('status')=='ok' and d['expected_found'] is not False and bool(d.get('content','').strip()) and d.get('finish_reason')=='stop' else 'incomplete_token_limit' if d.get('finish_reason')=='length' else 'not_verified'
                    return d
                if not worker.is_alive():return {'status':'error','error':'Chat worker exited'}
            return {'status':'skipped','reason':'Deadline reached during chat check'}
        finally:
            if worker.is_alive():worker.terminate()
            worker.join(timeout=.5);recv.close()
    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:os.killpg(self.proc.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:self.proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid,signal.SIGKILL);self.proc.wait(timeout=1)
        if self.log:self.log.close()
        self.proc=None;self.log=None;self.state.pop('trial_pid',None);self.state.pop('trial_alias',None)
        self.save_state()


def _chat_worker(conn,base,alias,prompt,timeout,max_tokens):
    try:
        d=http(base,'/v1/chat/completions',{'model':alias,'messages':[{'role':'user','content':prompt}],'temperature':0,'seed':42,'reasoning_effort':'low','max_tokens':max_tokens,'cache_prompt':False},timeout)
        choice=d['choices'][0];msg=choice['message']
        conn.send({'status':'ok','content':msg.get('content') or '', 'reasoning_content':msg.get('reasoning_content') or '', 'finish_reason':choice.get('finish_reason'),'usage':d.get('usage')})
    except Exception as e:conn.send({'status':'error','error':repr(e)})
    finally:conn.close()


class Service:
    def __init__(self,name,router,state,root):self.name=name;self.router=router.rstrip('/');self.state=state;self.root=pathlib.Path(root)
    def persist(self):
        tmp=self.root/'state.tmp';tmp.write_text(json.dumps(self.state,indent=2));tmp.replace(self.root/'state.json')
    def inspect(self,allow_busy=False):
        if self.name=='none':return {}
        active=subprocess.run(['systemctl','--user','is-active','--quiet',self.name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
        self.state['service_name']=self.name;self.state['service_was_active']=active;self.state['loaded_models']=[]
        if not active:return {}
        # Fail closed: do not stop an active service unless its workload can be checked.
        models=http(self.router,'/models',timeout=3)['data']
        for m in models:
            status=m.get('status',{}).get('value')
            if status not in ('loaded','unloaded') and not allow_busy:
                raise RuntimeError(f'Model {m["id"]} is {status}; wait until the server is idle')
            if status=='loaded':
                import urllib.parse
                slots=http(self.router,'/slots?model='+urllib.parse.quote(m['id'],safe=''),timeout=3)
                if (not isinstance(slots,list) or any(s.get('is_processing') for s in slots)) and not allow_busy:raise RuntimeError(f'Model {m["id"]} is busy; tuning was not started')
                self.state['loaded_models'].append(m['id'])
        pid=int(subprocess.check_output(['systemctl','--user','show',self.name,'--property=MainPID','--value'],text=True).strip())
        args=pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().decode().split('\0')[1:]
        return {k:v for k,v in parse_argv(args).items() if k in PARENT_KEYS}
    def pause(self):
        if not self.state.get('service_was_active'):return
        # Persist intent before stopping, so recover can handle an abrupt process death.
        self.state['service_paused']=True;self.persist()
        subprocess.run(['systemctl','--user','stop',self.name],check=True,timeout=10)
    def restore(self):
        if not self.state.get('service_paused'):return
        try:
            subprocess.run(['systemctl','--user','start',self.name],check=True,timeout=10)
            ready=False
            for _ in range(30):
                try:http(self.router,'/health',timeout=.2);ready=True;break
                except Exception:time.sleep(.1)
            self.state['restore_models_requested']=[]
            if ready:
                for name in self.state.get('loaded_models',[]):
                    try:
                        http(self.router,'/models/load',{'model':name},timeout=2)
                        self.state['restore_models_requested'].append(name)
                    except Exception as e:self.state.setdefault('restore_warnings',[]).append(str(e))
            self.state['service_paused']=False
        except Exception as e:self.state.setdefault('restore_warnings',[]).append(repr(e))
        self.persist()
