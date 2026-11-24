"""Independent CPU acceptance of exported Food500 files and original supervision.

No Parquet dependency: checks exported bytes/RGB, official text, ledger accounting,
split isolation and the existing Caption Dataset/Collator used by A/B/B-global.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import warnings

from PIL import Image
import torch
from foodlens.data import (CaptionManifestDataset,audit_caption_rows,collate_caption,
                           read_jsonl,safe_image_path,sha256,length_audit)
from foodlens.text import Vocabulary


def verify(args):
    torch.set_num_threads(2)
    rows = read_jsonl(args.manifest)
    counts = audit_caption_rows(rows)
    assert all(counts.values()),'empty split'
    summary = json.loads((args.evidence/'build_summary.json').read_text(encoding='utf-8'))
    lock = json.loads((args.evidence/'source_lock.json').read_text(encoding='utf-8'))
    assert sha256(args.manifest)==summary['manifest_sha256']
    official = {}
    for name,spec in lock['official_files'].items():
        path = args.data_root/'Food500-Cap'/name
        assert sha256(path) in {spec['sha256'],*spec.get('also_accepted_git_sha256',[])}
        parsed = json.loads(path.read_text(encoding='utf-8'))
        canonical=(json.dumps(parsed,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n').encode('utf-8')
        assert hashlib.sha256(canonical).hexdigest()==spec['canonical_records_sha256']
        for r in parsed:
            official[spec['split'],r['filename']] = r
    evidence = read_jsonl(args.evidence/'mapping_evidence.jsonl')
    exclusions = read_jsonl(args.evidence/'exclusions.jsonl')
    accepted = {r['id']:r for r in evidence if r['status']=='included'}
    assert set(accepted)=={r['id'] for r in rows}
    accounted = Counter((r['original_split'],r['official']['filename']) for r in evidence if r['matched'])
    accounted.update((r['official_split'],r['official']['filename']) for r in exclusions if r['reason']=='no_confirmed_community_image')
    assert set(accounted)==set(official) and all(v==1 for v in accounted.values()),'official ledger not exhaustive'
    assert len(evidence)==sum(summary['community_records'].values())
    by_image = {}
    isolation = {field:defaultdict(set) for field in ['sha256','rgb_sha256','duplicate_group','official_filename']}
    for r in rows:
        source = official[r['official_split'],r['official_filename']]
        assert r['caption']==source['caption'] and r['category']==source['cat'],'official text changed'
        assert r['caption'].strip() and isinstance(r['id'],str)
        ev = accepted[r['id']]
        assert ev['official']==source and ev['sha256']==r['sha256'] and ev['image']==r['image']
        assert ev['split']==r['split'] and ev['duplicate_group']==r['duplicate_group']
        assert r['source_revision']==lock['revision']
        assert (r['official_split']=='train' and r['split'] in {'train','val'}) or (r['official_split']=='test' and r['split']=='test')
        if r['image'] in by_image:
            assert by_image[r['image']]['sha256']==r['sha256']
        by_image[r['image']] = r
        for field in isolation:
            isolation[field][r[field]].add(r['split'])
    for field,groups in isolation.items():
        assert all(len(s)==1 for s in groups.values()),'cross-split '+field
    exports = {p.relative_to(args.data_root).as_posix() for p in (args.data_root/'caption_images').rglob('*') if p.is_file()}
    assert exports==set(by_image),'exported asset set differs from manifest images'
    inventory = read_jsonl(args.evidence/'image_inventory.jsonl')
    assert {r['image'] for r in inventory}==set(by_image)
    assert all(r['sha256']==by_image[r['image']]['sha256'] for r in inventory)
    def check_image(pair):
        relative,r = pair
        path = safe_image_path(args.data_root,relative)
        assert sha256(path)==r['sha256'],relative
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',UserWarning)
            with Image.open(path) as image:
                rgb=image.convert('RGB');rgb.load()
                value=hashlib.sha256(f'RGB:{rgb.width}x{rgb.height}\0'.encode('ascii')+rgb.tobytes()).hexdigest()
                assert value==r['rgb_sha256'],relative
        return value
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(check_image,by_image.items()))
    groups = json.loads((args.evidence/'duplicate_groups.json').read_text(encoding='utf-8'))
    for g in groups:
        if g['cross_original_split']:
            assert all(m['final_split'] is None and m['reason']=='test_priority_exact_duplicate'
                       for m in g['members'] if m['original_split']=='train')
            assert all(m['final_split']=='test' for m in g['members'] if m['original_split']=='test')
    vocab=Vocabulary()
    train=[r['caption'] for r in rows if r['split']=='train']
    vocab.extend_training(train,split='train')
    result={'passed':True,'counts':counts,'manifest_sha256':sha256(args.manifest),'unique_images_checked':len(by_image),
            'official_records_accounted':len(accounted),'unresolved_community_records':summary['unresolved_community_records'],
            'exclusions':dict(Counter(r['reason'] for r in exclusions)),'cross_split_sha256':0,'cross_split_rgb_sha256':0,
            'cross_split_duplicate_group':0,'cross_split_official_filename':0,'training_vocab_size':len(vocab),
            'training_vocab_sha256':hashlib.sha256(json.dumps(vocab.tokens,ensure_ascii=False).encode()).hexdigest(),
            'dataset_collator':{},'length_audit':{}}
    for split in ('train','val','test'):
        ds=CaptionManifestDataset(args.manifest,args.data_root,split=split)
        indices=sorted({0,len(ds)-1,max(range(len(ds)),key=lambda i:len(ds.rows[i]['caption']))})
        batch=collate_caption([ds[i] for i in indices],vocab,max_tokens=64)
        assert bool(torch.isfinite(batch['images']).all())
        result['dataset_collator'][split]={'ids':batch['ids'],'images':list(batch['images'].shape),'inputs':list(batch['inputs'].shape),
                                         'truncated':batch['truncated'],'families':['A','B','B-global'],'shared_manifest_sha256':sha256(args.manifest)}
        result['length_audit'][split]=length_audit([r['caption'] for r in ds.rows],vocab,64)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=Path('../data/caption_manifest.jsonl'))
    parser.add_argument('--data-root',type=Path,default=Path('../data'))
    parser.add_argument('--evidence',type=Path,default=Path('../outputs/food500_manifest_build'))
    parser.add_argument('--out',type=Path,default=Path('../outputs/food500_manifest_build/independent_audit.json'))
    verify(parser.parse_args())


if __name__=='__main__':
    main()
