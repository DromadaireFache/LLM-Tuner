import configparser
import os
import pathlib
import re
import struct


def duration(value):
    """Accept 600, 10m, 10min, 1hr, 8h, or 1h30m."""
    s=str(value).lower().replace(' ','')
    if s.isdigit(): return float(s)
    matches=list(re.finditer(r'(\d+(?:\.\d+)?)(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h|days?|d)',s))
    if not matches or ''.join(m.group() for m in matches)!=s:
        raise ValueError('Use a duration such as 10min, 1hr, 8h, or 1h30m')
    return sum(float(m[1])*({'s':1,'m':60,'h':3600,'d':86400}[m[2][0]]) for m in matches)


class GGUF:
    """Read metadata and tensor offsets without loading model weights or numpy."""
    scalar={0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
    def __init__(self,path):
        self.path=str(pathlib.Path(path).expanduser().resolve())
        self.meta={};self.tensors=[];self.size=0
        p=pathlib.Path(self.path)
        split=re.match(r'(.+)-\d{5}-of-(\d{5})\.gguf$',p.name)
        paths=sorted(p.parent.glob(split[1]+'-?????-of-'+split[2]+'.gguf')) if split else [p]
        if split and len(paths)!=int(split[2]):raise ValueError('Missing GGUF shards')
        for shard in paths:self._read(shard)
        self.arch=self.meta.get('general.architecture','unknown')
        self.layers=int(self.meta.get(self.arch+'.block_count',0))
        self.experts={}
        self.layer_bytes={}
        for t in self.tensors:
            m=re.match(r'blk\.(\d+)\.',t['name'])
            if m:
                il=int(m[1]);self.layer_bytes[il]=self.layer_bytes.get(il,0)+t['bytes']
                if '_exps' in t['name']:self.experts[il]=self.experts.get(il,0)+t['bytes']
        if not self.layers and self.layer_bytes:self.layers=max(self.layer_bytes)+1
        self.loop_start=int(self.meta.get(self.arch+'.loop.start',0))
        self.loop_end=int(self.meta.get(self.arch+'.loop.end',0))
        self.passes=int(self.meta.get(self.arch+'.loop.count',1))
        self.logical_layers=self.layers+(self.loop_end-self.loop_start)*max(0,self.passes-1)
    def _read(self,path):
        with open(path,'rb') as f:
            def read(fmt):
                n=struct.calcsize('<'+fmt);b=f.read(n)
                if len(b)!=n:raise ValueError('Truncated GGUF header: '+str(path))
                return struct.unpack('<'+fmt,b)[0]
            def string():
                n=read('Q')
                if n>1024**3:raise ValueError('Invalid GGUF string length')
                b=f.read(n)
                if len(b)!=n:raise ValueError('Truncated GGUF string')
                return b.decode('utf-8',errors='replace')
            def value(typ,keep=True):
                if typ in self.scalar:return read(self.scalar[typ])
                if typ==8:return string()
                if typ==9:
                    et=read('I');n=read('Q')
                    if n>100_000_000:raise ValueError('Invalid GGUF array length')
                    if not keep and et in self.scalar:
                        f.seek(n*struct.calcsize('<'+self.scalar[et]),1);return None
                    vals=[]
                    for _ in range(n):
                        v=value(et,keep)
                        if keep:vals.append(v)
                    return vals if keep else None
                raise ValueError('Unsupported GGUF metadata type '+str(typ))
            if f.read(4)!=b'GGUF':raise ValueError('Not a little-endian GGUF file: '+str(path))
            version=read('I')
            if version not in (2,3):raise ValueError('Unsupported GGUF version '+str(version))
            nt=read('Q');nk=read('Q')
            if nt>10_000_000 or nk>1_000_000:raise ValueError('Unreasonable GGUF header counts')
            local={}
            for _ in range(nk):
                key=string();typ=read('I');v=value(typ,not key.startswith('tokenizer.'))
                if v is not None:local[key]=v
            self.meta.update({k:v for k,v in local.items() if k not in self.meta})
            ts=[]
            for _ in range(nt):
                name=string();nd=read('I')
                if nd>16:raise ValueError('Invalid GGUF tensor dimensions')
                dims=[read('Q') for _ in range(nd)];typ=read('I');offset=read('Q')
                ts.append({'name':name,'dims':dims,'type':typ,'offset':offset,'shard':path.name})
            alignment=int(local.get('general.alignment',32));start=(f.tell()+alignment-1)//alignment*alignment
            size=path.stat().st_size;ordered=sorted(ts,key=lambda t:t['offset'])
            for i,t in enumerate(ordered):
                end=ordered[i+1]['offset'] if i+1<len(ordered) else size-start
                t['bytes']=max(0,end-t['offset'])
            self.tensors.extend(ts);self.size+=size
    def priority(self):
        # Repeated layers yield more saved CPU work per byte of VRAM.
        return sorted(self.experts,key=lambda il:(-(self.passes if self.loop_start<=il<self.loop_end else 1),self.experts[il],il))
    def summary(self):
        return {'path':self.path,'architecture':self.arch,'file_bytes':self.size,'physical_layers':self.layers,'logical_layers':self.logical_layers,'expert_layer_bytes':self.experts,'loop':{'start':self.loop_start,'end':self.loop_end,'passes':self.passes},'training_context':self.meta.get(self.arch+'.context_length')}


ALIASES={'ngl':'n-gpu-layers','gpu-layers':'n-gpu-layers','c':'ctx-size','b':'batch-size','ub':'ubatch-size','ctk':'cache-type-k','ctv':'cache-type-v','t':'threads','tb':'threads-batch','np':'parallel','fa':'flash-attn','lm':'load-mode','ctxcp':'ctx-checkpoints','swa-checkpoints':'ctx-checkpoints','ot':'override-tensor','ncmoe':'n-cpu-moe','cmoe':'cpu-moe'}
DROP={'host','port','alias','models-dir','models-preset','models-max','load-on-startup','log-file','log-colors','log-verbosity','verbosity','threads-http','api-key','api-key-file','ssl-key-file','ssl-cert-file','metrics','props','slots','slot-save-path','path','webui','no-webui','fit','timeout'}
PARENT_KEYS={'ctx-size','n-gpu-layers','batch-size','ubatch-size','threads','threads-batch','parallel','load-mode','flash-attn','cache-type-k','cache-type-v','cpu-moe','n-cpu-moe','no-kv-offload','no-op-offload','override-tensor','ctx-checkpoints','cache-ram'}


def canonical(k):
    k=k.lower().lstrip('-')
    if k.startswith('llama_arg_'):
        k=k[len('llama_arg_'):].replace('_','-')
        k={'batch':'batch-size','ubatch':'ubatch-size','ctx-checkpoints':'ctx-checkpoints','kv-offload':'kv-offload','op-offload':'op-offload'}.get(k,k)
    return ALIASES.get(k,k)


def read_presets(path):
    p=configparser.ConfigParser(interpolation=None,strict=False,inline_comment_prefixes=('#',';'))
    text=pathlib.Path(path).expanduser().read_text()
    # llama.cpp accepts top-level version=1, while ConfigParser needs a section.
    text=re.sub(r'(?m)^version\s*=.*\n','',text)
    p.read_string(text)
    glob=dict(p['*']) if '*' in p else {}
    return {s:{canonical(k):v.strip().strip('"') for k,v in (glob|dict(p[s])).items()} for s in p.sections() if s!='*'}


def expert_pattern(cpu_layers):
    ids=sorted(set(cpu_layers))
    return r'blk\.('+'|'.join(map(str,ids))+r')\.ffn_.*_exps.*=CPU' if ids else None


def placement(config,model,gpu_layers):
    c=dict(config)
    for k in ('cpu-moe','n-cpu-moe','override-tensor','moe-cache-mib','moe-cache-cpu-share'):c.pop(k,None)
    gpu=set(gpu_layers);cpu=set(model.experts)-gpu
    if not gpu:c['cpu-moe']=True
    elif cpu:c['override-tensor']=expert_pattern(cpu)
    return c


def as_bool(v):return v is True or str(v).lower() in ('true','yes')


def normalized(c,forced=None):
    d={canonical(k):v for k,v in c.items() if canonical(k) not in DROP}
    for k,v in list(d.items()):
        if isinstance(v,str) and v.lower() in ('true','false'):d[k]=v.lower()=='true'
    if forced:d.update(forced)
    d.setdefault('ubatch-size',128)
    d['batch-size']=max(int(d.get('batch-size',512)),int(d.get('ubatch-size',128)))
    for k in ('cpu-moe','no-kv-offload','no-op-offload'):
        if not as_bool(d.get(k,False)):d.pop(k,None)
    return d


def cli_args(config,supported):
    args=[]
    for key,value in sorted(config.items()):
        if key not in supported:continue
        if isinstance(value,bool):
            if value:args.append('--'+key)
            elif key.startswith('no-') and key[3:] in supported:args.append('--'+key[3:])
            elif 'no-'+key in supported:args.append('--no-'+key)
        else:args += ['--'+key,str(value)]
    return args


def supported_flags(help_text):
    return {canonical(k) for k in re.findall(r'--([a-z][a-z0-9-]*)',help_text)}


def parse_argv(args):
    d={};i=0
    while i<len(args):
        arg=args[i]
        if arg.startswith('-'):
            k=canonical(arg)
            if i+1<len(args) and (not args[i+1].startswith('-') or re.match(r'^-\d',args[i+1])):d[k]=args[i+1];i+=1
            else:d[k]=True
        i+=1
    return d
