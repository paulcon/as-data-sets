import json,sys,base64,os
nb=json.load(open(sys.argv[1])); out=sys.argv[2]; tag=sys.argv[3]
for i,c in enumerate(nb['cells']):
    k=0
    for o in c.get('outputs',[]):
        d=o.get('data',{})
        if 'image/png' in d:
            fn=f'{out}/{tag}_c{i}_{k}.png'; open(fn,'wb').write(base64.b64decode(d['image/png'])); k+=1
