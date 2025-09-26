import os
import json
import tqdm
import argparse

import random
random.seed(42)

import numpy as np


language_map = {
    'dsgs': {
        'source': 'deu',
        'target': 'sgg',
    },
    'lsf-ch': {
        'source': 'fra',
        'target': 'fsl',
    },
    'lis-ch': {
        'source': 'ita',
        'target': 'ise',
    },
}

parser = argparse.ArgumentParser()
parser.add_argument("--language", type=str, default='dsgs')
parser.add_argument("--num_eval", type=int, default=11)
parser.add_argument("--dsgs03", action='store_true')
parser.add_argument('--debug', action='store_true')
args = parser.parse_args()

text_path = './system_outputs/signsuisse_test/test.txt'
segments = {
    'ref': [],
    'sockeye': [],
    'sign_mt_v2': [],
}
system_prefix = {
    'ref': 20000,
    'sockeye': 30000,
    'sockeye_repeat': 31000,
    'sign_mt_v2': 40000,
    'sign_mt_v2_repeat': 41000,
}

with open('./human_evaluation/batches_text2pose/results/text2poseSignsuisse.deu-sgg.sockeye.id.txt', 'r') as f:
    annotated_sockeye_ids = [line.strip() for line in f.readlines()]

# Gather items
with open(text_path) as file:
    count = 0
    for index, line in tqdm.tqdm(enumerate(file)):           
        sent = line.rstrip()

        if sent.startswith(f'<{args.language}>'):
            if args.language == 'dsgs':
                # HACK: we first take from the last 250 segments for dsgs (in total 500)
                count = count + 1
                if count <= 250:
                    continue

            text = ' '.join(sent.split(' ')[1:])

            # 50 ref, 175 (150+25) sockeye, and 275 (250+25) sign_mt_v2
            for system in segments.keys():
                if args.language == 'dsgs':
                    if system == 'ref' and len(segments['ref']) == 50:
                        continue
                    # take 25 duplication from round 1 for sockeye
                    if system == 'sockeye' and len(segments['sockeye']) >= 150 and str(index) not in annotated_sockeye_ids[-25:]:
                        continue
                    # take 100 duplication from round 1 for sign_mt (v1)
                    # comment out the following since we can just use the last 250
                    # if system == 'sign_mt_v2' and len(segments['sign_mt_v2']) == 250:
                    #     continue
                    # if system == 'sign_mt_v2' and len(segments['sign_mt_v2']) >= 150 and str(index) not in annotated_sockeye_ids:
                    #     continue

                    # 25 ref and no sockeye for dsgs03
                    if args.dsgs03:
                        if system == 'ref' and len(segments['ref']) == 25:
                            continue
                        if system == 'sockeye':
                            continue
                else:
                    if system == 'ref' and len(segments['ref']) == 50:
                        continue

                video_path_remote = f'https://pub.cl.uzh.ch/projects/iict/signsuisse_test/{system}/{index}.mp4'

                segment = {
                    "_block": -1,
                    "_item": -1,
                    "documentID": -1,
                    "isCompleteDocument": False,
                    # "itemID": index,
                    "itemID": system_prefix[system] + index,
                    "itemType": "REF" if system == 'ref' else "TGT",
                    # "itemType": system,
                    "sourceContextLeft": "",
                    "sourceID": "signsuisse_test",
                    "sourceText": text,
                    "targetContextLeft": "",
                    "targetID": system,
                    "targetText": video_path_remote,
                }
                segments[system].append(segment)

# Repeat
selected_elements = [s.copy() for s in segments['sign_mt_v2'][-25:]]
for element in selected_elements:
    element['itemID'] = element['itemID'] - system_prefix['sign_mt_v2'] + system_prefix['sign_mt_v2_repeat']
segments['sign_mt_v2'].extend(selected_elements)

if args.language != 'dsgs':
    selected_elements = [s.copy() for s in segments['sockeye'][-25:]]
    for element in selected_elements:
        element['itemID'] = element['itemID'] - system_prefix['sockeye'] + system_prefix['sockeye_repeat']
    segments['sockeye'].extend(selected_elements)

# Mix
all_items = []
for system, items in segments.items():
    if len(items) == 0:
        continue

    print(f'system {system} has {len(items)} items')
    print(f'system {system} has {len(set([item["itemID"] for item in items]))} unique ids')
    print(f'from {items[0]["itemID"]} to {items[-1]["itemID"]}')
    print([item['itemID'] for item in items])
    all_items.extend(items)

print('In total:', len(all_items))
print('Unique itemIDs:', len(set([item['itemID'] for item in all_items])))
random.shuffle(all_items)

if args.debug:
    all_items = all_items[:100]

# Add document level items
document_size = 10
document_id_prefix = 1000000
modified_items = []
# Step 1: Add a document-level item after every 10 items and assign documentID
documentID = 0
for i in range(0, len(all_items), document_size):
    # Add the 10 items from the group
    group = all_items[i:i+document_size]
    
    # Assign documentID to each of the 10 items in the group
    for item in group:
        item['documentID'] = f"signsuisse.{args.language}.{documentID}"
    
    # Insert the document-level item after the group
    document_item = {
        "_block": -1,
        "_item": -1,
        "documentID": f"signsuisse.{args.language}.{documentID}",
        "isCompleteDocument": True,
        "itemID": document_id_prefix + documentID,
        "itemType": 'TGT',
        "sourceContextLeft": "",
        "sourceID": "signsuisse_test",
        "sourceText": "skip this",
        "targetContextLeft": "",
        "targetID": 'mix',
        "targetText": '',
    }
    
    # Append the group and the document item
    modified_items.extend(group)
    modified_items.append(document_item)
    
    # Increment the documentID for the next group
    documentID += 1
# Step 2: Assign each item a unique _item ID starting from 0
for index, item in enumerate(modified_items):
    item['_item'] = index

# print(modified_items)
# exit()

batch_size = 100
batches = []
current_batch_items = []

# Create batches
for item in modified_items * args.num_eval:
    current_batch_items.append(item)

    if len(current_batch_items) == batch_size + batch_size / document_size:
        batch = {
            "items": current_batch_items,
            "task": {
                "batchNo": len(batches) + 1,
                "batchSize": batch_size,
                "randomSeed": 1111,
                "requiredAnnotations": 1,
                "sourceLanguage": language_map[args.language]['source'],
                "targetLanguage": language_map[args.language]['target'],
            },
        }

        batches.append(batch)
        current_batch_items = []

output_path = \
f'''./human_evaluation/batches_text2pose/batches.text2pose.signsuisse.\
{language_map[args.language]["source"]}-{language_map[args.language]["target"]}.\
r2{".debug." if args.debug else "."}{"dsgs03." if args.dsgs03 else ""}{args.num_eval}.json'''

with open(output_path, 'w') as fp:
    json.dump(batches, fp, indent=2)
