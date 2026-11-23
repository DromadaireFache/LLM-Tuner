import collections
import json
import math
import random

from .model import normalized, placement


def key(config):return json.dumps(config,sort_keys=True,separators=(',',':'))


def gpu_experts(config,model):
    import re
    if config.get('cpu-moe'):return []
    try:ngl=int(config.get('n-gpu-layers',model.logical_layers+1))
    except (ValueError,TypeError):ngl=model.logical_layers+1
    start=max(0,model.logical_layers+1-ngl)
    resident={i for i in model.experts if i>=start}
    if 'override-tensor' in config:
        expr=config['override-tensor'].rsplit('=',1)[0]
        try:return [i for i in model.experts if i in resident and not re.search(expr,f'blk.{i}.ffn_down_exps.weight')]
        except re.error:return []
    n=int(config.get('n-cpu-moe',0))
    return [i for i in model.experts if i in resident and i>=n]


class Search:
    """Interleave parameter families, then refine measured winners; never enumerate 2**layers."""
    def __init__(self,model,base,capacities,hardware,supported,forced=None,seed=42):
        self.model=model;self.supported=supported;self.forced=forced or {};self.rng=random.Random(seed)
        self.seen=set();self.queued=set();self.groups=collections.OrderedDict((k,collections.deque()) for k in ('baseline','batch','placement','gpu','threads','kv','operation','loading','cache','adaptive'))
        self.base=normalized(base,self.forced)
        self.capacities=capacities;self.best=[];self.counter=0
        allgpu=model.logical_layers+1 or 999
        logical=int(hardware.get('logical_cpus',1));physical=int(hardware.get('physical_cpus',logical))
        self.add('baseline',self.base)
        common=self.base|{'n-gpu-layers':allgpu,'batch-size':512,'ubatch-size':128,'parallel':1,'cache-ram':256,'ctx-checkpoints':1}
        common.pop('no-kv-offload',None)
        if model.experts:
            common=placement(common,model,[]);common['no-op-offload']=True
        # Batch variants precede arbitrary placements so short budgets still investigate prefill.
        for ub in (64,128,256,32,512,16):self.add('batch',common|{'ubatch-size':ub,'batch-size':max(512,ub)})
        if model.experts:
            rank=model.priority();n=len(rank)
            nonexpert=max(0,model.size-sum(model.experts.values()))
            free=hardware.get('gpu_free_mib',0)*1024**2
            rough=max(0,free-nonexpert-2*1024**3)
            avg=sum(model.experts.values())/max(1,n)
            est=min(n,max(1,int(rough/max(1,avg))))
            counts=list(dict.fromkeys([1,2,est,max(1,est//2),min(n,est+1),min(n,est+2),n//2,n]))
            for count in counts:
                for order in (rank,list(reversed(rank)),sorted(rank)):
                    picked=order[:count]
                    for ub in (128,64):self.add('placement',placement(common,model,picked)|{'ubatch-size':ub})
            for ncpu in sorted(set([0,n//4,n//2,3*n//4,n])):
                c=placement(common,model,[]);c.pop('cpu-moe',None);c['n-cpu-moe']=ncpu
                self.add('placement',c)
        # Partial attention/dense-layer offload is useful for dense models and KV-heavy architectures.
        for ngl in dict.fromkeys([allgpu,max(1,allgpu*3//4),max(1,allgpu//2),max(1,allgpu//4),0]):
            for ub in (128,64):self.add('gpu',common|{'n-gpu-layers':ngl,'ubatch-size':ub})
        for t in dict.fromkeys([logical,physical,max(1,physical//2)]):
            self.add('threads',common|{'threads':t})
            self.add('threads',common|{'threads-batch':t})
        for types in [('q8_0','q8_0'),('q4_0','q8_0'),('q4_0','q4_0'),('f16','f16')]:
            for cpu in (False,True):self.add('kv',common|{'cache-type-k':types[0],'cache-type-v':types[1],'no-kv-offload':cpu})
        for ub in (64,128,32):self.add('operation',common|{'no-op-offload':False,'ubatch-size':ub})
        self.add('kv',common|{'flash-attn':'off','cache-type-k':'f16','cache-type-v':'f16'})
        for mode in ('mmap','none'):self.add('loading',common|{'load-mode':mode})
        if model.experts and 'moe-cache-mib' in supported:
            for mib in (512,1024,1536):
                for share in (0,0.5,1):self.add('cache',placement(common,model,[])|{'moe-cache-mib':mib,'moe-cache-cpu-share':share,'ubatch-size':32})
        for cap in capacities:
            if cap!=int(common.get('ctx-size',0)):self.add('gpu',common|{'ctx-size':cap})
        self.family_index=0
    def add(self,family,c,front=False):
        c=normalized(c,self.forced)
        c={k:v for k,v in c.items() if k in self.supported}
        k=key(c)
        if k in self.seen or k in self.queued:return
        self.queued.add(k)
        if front:self.groups[family].appendleft(c)
        else:self.groups[family].append(c)
    def next(self):
        names=list(self.groups)
        for _ in names:
            name=names[self.family_index%len(names)];self.family_index+=1
            if self.groups[name]:
                c=self.groups[name].popleft();k=key(c);self.queued.discard(k);self.seen.add(k);return c,name
        # Longer budgets explore weighted random expert subsets around the best measured placements.
        if self.best:
            # A duplicate random draw must not prematurely end a long-budget search.
            for _ in range(64):
                c=dict(self.rng.choice(self.best[:4])[1])
                if self.model.experts:
                    gpu=set(gpu_experts(c,self.model));ids=list(self.model.experts)
                    for _ in range(self.rng.choice([1,1,2,3])):
                        il=self.rng.choice(ids)
                        if il in gpu:gpu.remove(il)
                        else:gpu.add(il)
                    c=placement(c,self.model,gpu)
                else:
                    try:ngl=int(c.get('n-gpu-layers',self.model.logical_layers+1))
                    except ValueError:ngl=self.model.logical_layers+1
                    c['n-gpu-layers']=max(0,min(self.model.logical_layers+1,ngl+self.rng.choice([-2,-1,0,1,2])))
                c['ubatch-size']=self.rng.choice([32,64,96,128,192,256,384,512])
                c['threads-batch']=self.rng.choice([max(1,int(self.base.get('threads',1))//2),int(self.base.get('threads',1))])
                c['ctx-size']=self.rng.choice(self.capacities)
                self.add('adaptive',c)
                if self.groups['adaptive']:return self.next()
        return None,None
    def feedback(self,c,result):
        if result.get('status')!='ok':
            if int(c.get('ubatch-size',128))>16:self.add('adaptive',c|{'ubatch-size':max(16,int(c.get('ubatch-size',128))//2)},True)
            self.add('adaptive',c|{'no-kv-offload':True},True)
            self.add('adaptive',c|{'no-op-offload':True},True)
            if self.model.experts:
                gpu=set(gpu_experts(c,self.model))
                if gpu:
                    weakest=min(gpu,key=lambda il:(self.model.passes if self.model.loop_start<=il<self.model.loop_end else 1,-self.model.experts[il]))
                    self.add('adaptive',placement(c,self.model,gpu-{weakest}),True)
            else:
                try:ngl=min(int(c.get('n-gpu-layers',self.model.logical_layers+1)),self.model.logical_layers+1)
                except ValueError:ngl=self.model.logical_layers+1
                if ngl>0:self.add('adaptive',c|{'n-gpu-layers':max(0,ngl-max(1,ngl//4))},True)
            return
        t=result['timings'];cost=512/t['prompt_per_second']+128/t['predicted_per_second']
        self.best.append((cost,c));self.best.sort(key=lambda x:x[0]);self.best=self.best[:12]
        # Logical batch size and physical micro-batch are independent knobs.
        for b in (256,512,1024,2048):self.add('adaptive',c|{'batch-size':b})
        for mode in ('mmap','none'):self.add('adaptive',c|{'load-mode':mode})
        if self.model.experts:
            gpu=set(gpu_experts(c,self.model))
            missing=[il for il in self.model.priority() if il not in gpu]
            for il in missing[:2]:self.add('adaptive',placement(c,self.model,gpu|{il}))
            if gpu:self.add('adaptive',placement(c,self.model,gpu-{self.rng.choice(sorted(gpu))}))


def finalists(trials,limit=3):
    good=[t for t in trials if t.get('status')=='ok' and t.get('stage')=='screen']
    if not good:return []
    pp=max(good,key=lambda t:t['timings']['prompt_per_second'])
    tg=max(good,key=lambda t:t['timings']['predicted_per_second'])
    overall=sorted(good,key=lambda t:512/t['timings']['prompt_per_second']+128/t['timings']['predicted_per_second'])
    out=[]
    for t in [overall[0],pp,tg]+overall:
        if t['config'] not in out:out.append(t['config'])
        if len(out)>=limit:break
    return out
