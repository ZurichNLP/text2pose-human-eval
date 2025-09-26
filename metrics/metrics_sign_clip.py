import os
import sys
import argparse

import numpy as np
import pandas as pd
import tqdm

from pose_format import Pose
from pose_format.utils.generic import reduce_holistic

sys.path.append('/home/zifjia/fairseq/examples/MMPT')
from demo_sign import embed_pose, embed_text

def get_pose(pose_path: str):
    with open(pose_path, 'rb') as f:
        pose = Pose.read(f.read())
    return pose

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Only evaluate the first 5 samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for embedding computation')
    args = parser.parse_args()

    text_path = './system_outputs/signsuisse_test/test.txt'
    pose_dirs = {
        'ref': './system_outputs/signsuisse_test/ref/',
        'sockeye': './system_outputs/signsuisse_test/sockeye/',
        'sign_mt': './system_outputs/signsuisse_test/sign_mt/',
        'sign_mt_v2': './system_outputs/signsuisse_test/sign_mt_v2/',
    }
    systems = list(pose_dirs.keys())  # include 'ref'

    language_map = {
        'dsgs': {'source': 'deu', 'target': 'sgg'},
        'lsf-ch': {'source': 'fra', 'target': 'fsl'},
        'lis-ch': {'source': 'ita', 'target': 'ise'},
    }
    iso2_map = {'deu': 'de', 'fra': 'fr', 'ita': 'it'}
    model_names = ['default', 'suisse']

    # read lines
    with open(text_path) as f:
        lines = f.readlines()
    if args.debug:
        lines = lines[:5]

    # storage for reference
    ref_poses, ref_texts, ref_valid_idx = [], [], []
    ref_lengths = []
    # stats container
    all_stats = {}

    # --- process ref ---
    print("Preparing ref...")
    for idx, sent in enumerate(lines):
        sent = sent.rstrip()
        if not sent.startswith('<'): continue
        tag_end = sent.find('>')
        lang = sent[1:tag_end]
        if lang not in language_map: continue
        # text prompt
        src_iso3 = language_map[lang]['source']
        tgt_iso3 = language_map[lang]['target']
        src_iso2 = iso2_map.get(src_iso3, src_iso3[:2])
        prompt = f"<{src_iso2}> <{tgt_iso3}> {sent[tag_end+1:].strip()}"
        # pose
        ref_path = f"{pose_dirs['ref']}{idx}.raw.pose"
        if not os.path.exists(ref_path): continue
        p = get_pose(ref_path)
        ref_poses.append(p)
        ref_texts.append(prompt)
        ref_valid_idx.append(idx)
        ref_lengths.append(p.body.data.shape[0])

    # print(ref_texts)

    # compute ref stats
    arr = np.array(ref_lengths)
    all_stats['ref'] = {
        'count': len(arr),
        'min': int(arr.min()),
        'max': int(arr.max()),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'std': float(arr.std()),
    }
    print(f"Ref stats: {all_stats['ref']}")

    # embed ref once
    ref_pose_embeds = {m: [] for m in model_names}
    ref_text_embeds = {m: [] for m in model_names}
    for m in model_names:
        for i in tqdm.trange(0, len(ref_poses), args.batch_size, desc=f"Embedding ref poses ({m})", leave=False):
            ref_pose_embeds[m].extend(embed_pose(ref_poses[i:i+args.batch_size], model_name=m))
        for i in tqdm.trange(0, len(ref_texts), args.batch_size, desc=f"Embedding ref texts ({m})", leave=False):
            ref_text_embeds[m].extend(embed_text(ref_texts[i:i+args.batch_size], model_name=m))

    # --- process each system, including ref ---
    for system in systems:
        print(f"Processing {system}...")
        if system == 'ref':
            # score P-T for ref only
            entries = []
            for i, idx in enumerate(ref_valid_idx):
                lang = lines[idx].split(' ')[0][1:-1]
                entry = {
                    'data': 'signsuisse_test', 'system': 'ref', 'example_id': idx,
                    'source': language_map[lang]['source'], 'target': language_map[lang]['target']
                }
                for m in model_names:
                    # leave P-P empty
                    entry[f'SignCLIP{"" if m=="default" else "-Suisse"}-P-P'] = ''
                    # compute P-T with ref as hyp
                    ref_p = ref_pose_embeds[m][i]
                    txt_e = ref_text_embeds[m][i]
                    entry[f'SignCLIP{"" if m=="default" else "-Suisse"}-P-T'] = float(np.dot(ref_p, txt_e))
                entries.append(entry)
            pd.DataFrame(entries).to_csv(f'./metrics/metrics.ref.sign_clip.csv', index=False)
            continue

        # for other systems
        hyp_poses, hyp_texts, valid_idx = [], [], []
        for idx, sent in enumerate(lines):
            sent = sent.rstrip()
            if not sent.startswith('<'): continue
            lang = sent[1:sent.find('>')]
            if lang not in language_map: continue
            hyp_path = (
                f"{pose_dirs[system]}{idx}.imputed.pose" if system=='sockeye'
                else f"{pose_dirs[system]}{idx}.raw.pose"
            )
            if not os.path.exists(hyp_path): continue
            p = get_pose(hyp_path)
            hyp_poses.append(p)
            valid_idx.append(idx)
            src_iso3 = language_map[lang]['source']
            tgt_iso3 = language_map[lang]['target']
            src_iso2 = iso2_map.get(src_iso3, src_iso3[:2])
            hyp_texts.append(f"<{src_iso2}> <{tgt_iso3}> {sent[sent.find('>')+1:].strip()}")

        # print(hyp_texts)

        # hyp stats
        hl = np.array([p.body.data.shape[0] for p in hyp_poses])
        all_stats[system] = {
            'count': len(hl), 'min': int(hl.min()), 'max': int(hl.max()),
            'mean': float(hl.mean()), 'median': float(np.median(hl)), 'std': float(hl.std())
        }
        print(f"{system} stats: {all_stats[system]}")

        # embed hyp poses/texts
        hyp_pose_embeds = {m: [] for m in model_names}
        hyp_text_embeds = {m: [] for m in model_names}
        for m in model_names:
            for i in tqdm.trange(0, len(hyp_poses), args.batch_size, desc=f"Embedding {system} poses ({m})", leave=False):
                hyp_pose_embeds[m].extend(embed_pose(hyp_poses[i:i+args.batch_size], model_name=m))
            for i in tqdm.trange(0, len(hyp_texts), args.batch_size, desc=f"Embedding {system} texts ({m})", leave=False):
                hyp_text_embeds[m].extend(embed_text(hyp_texts[i:i+args.batch_size], model_name=m))

        # scoring
        entries = []
        for i, idx in enumerate(valid_idx):
            lang = lines[idx].split(' ')[0][1:-1]
            entry = {'data': 'signsuisse_test', 'system': system, 'example_id': idx,
                     'source': language_map[lang]['source'], 'target': language_map[lang]['target']}
            for m in model_names:
                ref_p = ref_pose_embeds[m][i]
                hyp_p = hyp_pose_embeds[m][i]
                txt_e = hyp_text_embeds[m][i]
                entry[f'SignCLIP{"" if m=="default" else "-Suisse"}-P-P'] = float(np.dot(ref_p, hyp_p))
                entry[f'SignCLIP{"" if m=="default" else "-Suisse"}-P-T'] = float(np.dot(hyp_p, txt_e))
            entries.append(entry)
        pd.DataFrame(entries).to_csv(f'./metrics/metrics.{system}.sign_clip.csv', index=False)

    # --- print combined table ---
    stats_df = pd.DataFrame.from_dict(all_stats, orient='index')
    stats_df.index.name = 'system'
    stats_df = stats_df.reset_index()
    print("\nCombined pose-length statistics:")
    print(stats_df.to_string(index=False))
    print("Done.")
