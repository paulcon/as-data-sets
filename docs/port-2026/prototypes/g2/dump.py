import json,sys
nb=json.load(open(sys.argv[1]))
print(nb.get('metadata',{}).get('kernelspec'))
for i,c in enumerate(nb['cells']):
    src=''.join(c['source'])
    print(f"\n##### [{i}] {c['cell_type']} exec={c.get('execution_count')}")
    print(src)
    for o in c.get('outputs',[]):
        if 'text' in o: print('--OUT text:', ''.join(o['text'])[:1500])
        elif 'data' in o:
            d=o['data']; ks=list(d.keys())
            if 'text/plain' in d: print('--OUT', ks, ''.join(d['text/plain'])[:1500])
            else: print('--OUT', ks)
        elif o.get('output_type')=='error': print('--ERR', o['ename'], o['evalue'])
