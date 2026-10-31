import json,sys
nb=json.load(open(sys.argv[1]))
print("kernel:",nb['metadata'].get('kernelspec'))
for i,c in enumerate(nb['cells']):
    print(f"\n===== cell {i} [{c['cell_type']}] =====")
    print(''.join(c['source']))
    for o in c.get('outputs',[]):
        if 'text' in o: print("--out--\n"+''.join(o['text'])[:3000])
        elif 'data' in o:
            d=o['data']
            if 'text/plain' in d: print("--out--\n"+''.join(d['text/plain'])[:3000])
            print("--data keys:",list(d.keys()))
        elif o.get('output_type')=='error': print("--ERR--",o.get('ename'),o.get('evalue'))
