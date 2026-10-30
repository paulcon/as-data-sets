import json, sys
nb = json.load(open(sys.argv[1]))
print("KERNEL", nb.get('metadata',{}).get('kernelspec'), "nbformat", nb.get('nbformat'))
for i, c in enumerate(nb['cells']):
    src = ''.join(c['source'])
    if c['cell_type'] == 'markdown':
        print(f"--- [{i}] MD\n{src}")
    else:
        print(f"--- [{i}] CODE\n{src}")
        for o in c.get('outputs', []):
            if 'text' in o:
                t=''.join(o['text']); print("  >>OUT:", t[:600])
            elif 'data' in o and 'text/plain' in o['data']:
                t=''.join(o['data']['text/plain']); print("  >>OUT:", t[:600])
            elif o.get('output_type')=='error':
                print("  >>ERR:", o.get('ename'), o.get('evalue'))
