import json
import csv
from shutil import copyfile
lst=[]
labelist=["contradiction", "neutral", "entailment"]
with open('train.jsonl','r') as f ,open('train-labels.lst') as fl:

    for l in f.readlines():
        item=json.loads(l)
        item['label']=int(fl.readline())
        lst.append(item)
    print(len(lst))
with open('train.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(
        ['index', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse',
         'sentence2_parse', 'sentence1', 'sentence2', 'label1', 'gold_label'])
    itemid=0
    for l in lst[1:]:
        s1=l['obs1']+'[SEP]'+l['obs2']
        if s1.__contains__('\n'):
            print('error')
            s1=s1.replace('\n', '')
        s2=l['hyp1']+'[SEP]'+l['hyp2']
        if s2.__contains__('\n'):
            print('error')
            s2=s2.replace('\n','')
        tsv_writer.writerow(
            [itemid, 'n', 'n', 'n', 'n', 'n',
             'n',
             'n', s1, s2, labelist[l['label']], labelist[l['label']]])
    out_file.close()

lst=[]
with open('dev.jsonl','r') as f ,open('dev-labels.lst') as fl:

    for l in f.readlines():
        item=json.loads(l)
        item['label']=int(fl.readline())
        lst.append(item)
    print(len(lst))
with open('dev_mismatched.tsv', 'wt') as out_file:
    tsv_writer = csv.writer(out_file, delimiter='\t')
    tsv_writer.writerow(
        ['index', 'promptID', 'pairID', 'genre', 'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse',
         'sentence2_parse', 'sentence1', 'sentence2', 'label1', 'label2', 'label3', 'label4', 'label5', 'gold_label'])
    itemid = 0
    for l in lst[1:]:
        s1=l['obs1']+'[SEP]'+l['obs2']
        if s1.__contains__('\n'):
            print('error')
            s1=s1.replace('\n', '')
        s2=l['hyp1']+'[SEP]'+l['hyp2']
        if s2.__contains__('\n'):
            print('error')
            s2=s2.replace('\n','')
        tsv_writer.writerow(
            [itemid, 'n', 'n', 'n', 'n', 'n',
             'n',
             'n', s1, s2, labelist[l['label']], labelist[l['label']], labelist[l['label']], labelist[l['label']], labelist[l['label']], labelist[l['label']]])
    out_file.close()
copyfile('dev_mismatched.tsv', 'dev_matched.tsv')

