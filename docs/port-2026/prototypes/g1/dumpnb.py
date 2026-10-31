import json,sys
nb=json.load(open(sys.argv[1]))
print("kernel:",nb.get('metadata',{}).get('kernelspec'))
for i,c in enumerate(nb['cells']):
    print(f"===== cell {i} [{c['cell_type']}] exec={c.get('execution_count')}")
    print(''.join(c['source']))
    for o in c.get('outputs',[]):
        if 'text' in o: print('--out(stream):'); print(''.join(o['text'])[:3000])
        d=o.get('data',{})
        if 'text/plain' in d: print('--out(plain):'); print(''.join(d['text/plain'])[:3000])
        if 'image/png' in d: print('--out: [image/png]')
        if o.get('output_type')=='error': print('--ERROR', o.get('ename'), o.get('evalue'))
