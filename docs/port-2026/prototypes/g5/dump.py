import json,sys
nb=json.load(open(sys.argv[1]))
print("kernel:",nb['metadata'].get('kernelspec'))
for i,c in enumerate(nb['cells']):
    print(f"==== cell {i} [{c['cell_type']}] exec={c.get('execution_count')}")
    print(''.join(c['source']))
    for o in c.get('outputs',[]):
        if 'text' in o: print('--out--'); print(''.join(o['text'])[:3000])
        elif 'data' in o:
            d=o['data']
            if 'text/plain' in d: print('--out(plain)--'); print(''.join(d['text/plain'])[:1500])
            if 'image/png' in d: print('--[png]--')
        if o.get('output_type')=='error': print('--ERROR--',o.get('ename'),o.get('evalue'))
