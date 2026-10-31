import json,sys,base64,os
for n in ['Ebola','HIV','MHD']:
    nb=json.load(open(f'/Users/paulconstantine/PycharmProjects/as-data-sets/{n}/{n}.ipynb'))
    for i,c in enumerate(nb['cells']):
        k=0
        for o in c.get('outputs',[]):
            d=o.get('data',{})
            if 'image/png' in d:
                open(f'img/{n}_c{i}_{k}.png','wb').write(base64.b64decode(d['image/png'])); k+=1
