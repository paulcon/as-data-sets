import json,sys
nb=json.load(open(sys.argv[1]))
print("kernel:",nb.get('metadata',{}).get('kernelspec'))
for i,c in enumerate(nb['cells']):
    src=''.join(c['source'])
    print(f"===== cell {i} [{c['cell_type']}] =====")
    print(src)
    if c['cell_type']=='code':
        for o in c.get('outputs',[]):
            if 'text' in o: print('--OUT--'); print(''.join(o['text'])[:3000])
            elif 'data' in o:
                d=o['data']
                if 'text/plain' in d: print('--OUT plain--'); print(''.join(d['text/plain'])[:2000])
                if 'image/png' in d: print('--OUT image/png--')
            elif o.get('output_type')=='error': print('--ERR--',o.get('ename'),o.get('evalue'))
