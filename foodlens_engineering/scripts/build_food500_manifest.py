"""Offline, content-addressed Food500 export. Never joins records by row order.

Run from foodlens_engineering: python -m scripts.build_food500_manifest --help
Source lock is checked before any export. Original JSON/Parquet are read-only.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path, PurePosixPath
import random
import sys
import warnings

LOCK = Path(__file__).with_name('food500_source_lock.json')
VERSION = 'foodlens.food500.build.v1'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode('utf-8')


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded(value))


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as stream:
        for row in rows:
            stream.write(encoded(row))


def load_official(path, spec):
    """Accept pinned local/Git serialization only when all parsed records agree."""
    actual_hash = file_hash(path)
    allowed = {spec['sha256'], *spec.get('also_accepted_git_sha256', [])}
    if actual_hash not in allowed:
        raise ValueError('official annotation byte hash mismatch: '+Path(path).name)
    rows = json.loads(Path(path).read_text(encoding='utf-8'))
    if len(rows) != spec['records'] or digest(encoded(rows)) != spec['canonical_records_sha256']:
        raise ValueError('official annotation canonical content mismatch: '+Path(path).name)
    return rows


def normalized_path(text):
    from unidecode import unidecode
    path = PurePosixPath(text.replace('\\', '/'))
    if path.is_absolute() or '..' in path.parts or ':' in text:
        raise ValueError('unsafe source filename')
    return unidecode(path.as_posix()).casefold()


def official_id(split, row):
    return 'food500_'+split+'_'+digest(encoded([row['filename'], row['caption']]))[:24]


def match_record(split, community, index):
    """Filename must be unique before caption filtering: never resolve a collision by first match."""
    from unidecode import unidecode
    image_path = community['image']['path']
    if not isinstance(image_path, str) or not image_path.strip():
        return None, 'missing_image_identifier', []
    name = community['cat']+'/'+image_path
    candidates = index.get((split, normalized_path(name)), [])
    if len(candidates) != 1:
        return None, 'ambiguous_path' if candidates else 'unmatched_path', candidates
    original = candidates[0]
    if unidecode(original['cat']).casefold() != unidecode(community['cat']).casefold():
        return None, 'category_mismatch', candidates
    if unidecode(original['caption']) != unidecode(community['caption']):
        return None, 'different_caption_requires_review', candidates
    if (original['filename'] == name and original['cat'] == community['cat']
            and original['caption'] == community['caption']):
        rule = 'exact'
    elif original['filename'] == name:
        rule = 'same_path_full_caption_unidecode'
    elif original['filename'].casefold() == name.casefold():
        rule = 'case_path_full_caption_unidecode'
    else:
        rule = 'transliterated_path_full_caption_unidecode'
    return original, rule, candidates


def duplicate_components(records):
    """Union by both byte hash and RGB hash, including transitive relations."""
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for field in ('sha256', 'rgb_sha256'):
        seen = {}
        for i, row in enumerate(records):
            previous = seen.setdefault(row[field], i)
            left, right = find(i), find(previous)
            if left != right:
                parent[max(left, right)] = min(left, right)
    grouped = defaultdict(list)
    for i, row in enumerate(records):
        grouped[find(i)].append(row)
    groups = []
    for members in grouped.values():
        group = 'dg_'+min(r['sha256'] for r in members)
        for row in members:
            row['duplicate_group'] = group
        groups.append((group, members))
    return sorted(groups)


def assign_splits(groups, seed=42):
    """Test owns its entire component, even if a Test mapping is unresolved.

    Only accepted original-Train components without any original-Test row enter
    Validation sampling. No test captions or scores are used for stratification.
    """
    strata = defaultdict(list)
    for group, members in groups:
        test = [r for r in members if r['original_split'] == 'test']
        for row in members:
            if not row.get('matched'):
                continue
            if test and row['original_split'] == 'train':
                row['exclusion_reason'] = 'test_priority_exact_duplicate'
            elif row['original_split'] == 'test':
                row['split'] = 'test'
        train = [r for r in members if r.get('matched') and r['original_split'] == 'train' and not test]
        if train:
            # Deterministic primary stratum for rare multi-category duplicate groups.
            category = min(r['official']['cat'] for r in train)
            strata[category].append((group, train))
    rng = random.Random(seed)
    audit = []
    for category, entries in sorted(strata.items()):
        entries.sort(key=lambda pair: pair[0])
        rng.shuffle(entries)
        count = max(1, min(len(entries)-1, int(len(entries)*.1+.5))) if len(entries) > 1 else 0
        validation = {key for key, _ in entries[:count]}
        for group, members in entries:
            for row in members:
                row['split'] = 'val' if group in validation else 'train'
        audit.append({'primary_category': category, 'train_side_groups': len(entries), 'val_groups': count})
    return audit


def build(args):
    import pyarrow.parquet as pq
    from PIL import Image
    if importlib.metadata.version('Unidecode') != '1.4.0':
        raise ValueError('mapping requires pinned Unidecode 1.4.0')
    root, out = args.data_root.resolve(), args.out.resolve()
    lock = json.loads(args.source_lock.read_text(encoding='utf-8'))
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = root/'Food500-Cap/Food500Cap-HF/data'
    expected = lock['shards']
    actual = {p.name for p in raw_dir.glob('*.parquet') if not p.name.startswith('._')}
    if actual != set(expected):
        raise ValueError('Parquet shard set differs from source lock')
    inputs = []
    for name, spec in sorted(expected.items()):
        path = raw_dir/name
        actual_hash = file_hash(path)
        if actual_hash != spec['sha256'] or path.stat().st_size != spec['size']:
            raise ValueError('source shard hash/size mismatch: '+name)
        inputs.append({'path':path.relative_to(root).as_posix(),'sha256':actual_hash,'bytes':path.stat().st_size})
    official, index = {}, defaultdict(list)
    for name, spec in sorted(lock['official_files'].items()):
        path = root/'Food500-Cap'/name
        rows = load_official(path, spec)
        for row in rows:
            key = (spec['split'], row['filename'])
            if key in official:
                raise ValueError('duplicate official filename within split')
            official[key] = row
            index[(spec['split'], normalized_path(row['filename']))].append(row)
        inputs.append({'path':path.relative_to(root).as_posix(),'sha256':file_hash(path),'records':len(rows)})
    records, matched_keys, warning_rows = [], set(), []
    image_inventory = {}
    for name, spec in sorted(expected.items()):
        print('Reading '+name, flush=True)
        for batch in pq.ParquetFile(raw_dir/name).iter_batches(batch_size=32):
            for community in batch.to_pylist():
                raw = community['image']['bytes']
                original, rule, candidates = match_record(spec['split'], community, index)
                h = digest(raw)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    with Image.open(io.BytesIO(raw)) as image:
                        fmt = image.format
                        rgb = image.convert('RGB')
                        rgb.load()
                        rgb_hash = digest(f'RGB:{rgb.width}x{rgb.height}\0'.encode('ascii')+rgb.tobytes())
                        dimensions = list(rgb.size)
                    warning_rows.extend({'shard':name,'image_path':community['image']['path'],'warning':str(w.message)} for w in caught)
                record = {'original_split':spec['split'],'community':{'image_path':community['image']['path'],'cat':community['cat'],'caption':community['caption']},
                          'shard':name,'sha256':h,'rgb_sha256':rgb_hash,'matched':original is not None,'mapping_rule':rule,
                          'path_candidate_count':len(candidates),'candidate_official_filenames':[r['filename'] for r in candidates]}
                if original is not None:
                    key = (spec['split'],original['filename'])
                    if key in matched_keys:
                        raise ValueError('multiple community records map to same official record')
                    matched_keys.add(key)
                    record['official'] = original
                    record['id'] = official_id(spec['split'],original)
                    # Some source .jpg names actually hold MPO (multi-picture JPEG).
                    # Keep original bytes, export .mpo, and decode primary frame just
                    # like the existing Dataset's Image.open(...).convert('RGB').
                    suffix = {'JPEG':'.jpg','MPO':'.mpo','PNG':'.png','WEBP':'.webp'}.get(fmt)
                    if suffix is None:
                        raise ValueError('unsupported image format: '+str(fmt))
                    relative = PurePosixPath('caption_images/sha256')/h[:2]/(h+suffix)
                    destination = root/relative
                    destination.parent.mkdir(parents=True,exist_ok=True)
                    if destination.exists():
                        if file_hash(destination) != h:
                            raise ValueError('existing export differs: '+relative.as_posix())
                    else:
                        with destination.open('xb') as stream:
                            stream.write(raw)
                    record['image'] = relative.as_posix()
                    image_inventory[h] = {'image':relative.as_posix(),'sha256':h,'rgb_sha256':rgb_hash,'bytes':len(raw),'format':fmt,'size':dimensions}
                else:
                    record['exclusion_reason'] = rule
                records.append(record)
    groups = duplicate_components(records)
    strata = assign_splits(groups,seed=args.seed)
    manifest = []
    for row in records:
        if 'split' not in row:
            continue
        original = row['official']
        manifest.append({'id':row['id'],'image':row['image'],'caption':original['caption'],'official_filename':original['filename'],
                         'sha256':row['sha256'],'duplicate_group':row['duplicate_group'],'split':row['split'],
                         'official_split':row['original_split'],'category':original['cat'],'rgb_sha256':row['rgb_sha256'],
                         'mapping_rule':row['mapping_rule'],'source_revision':lock['revision']})
    manifest.sort(key=lambda r: (r['split'],r['id']))
    if set(r['split'] for r in manifest) != {'train','val','test'} or len({r['id'] for r in manifest}) != len(manifest):
        raise ValueError('manifest must have three nonempty splits and unique IDs')
    payload = b''.join(encoded(row) for row in manifest)
    # Never silently replace a previously frozen manifest with a different result.
    if args.manifest.exists() and args.manifest.read_bytes() != payload:
        raise ValueError('manifest already exists with different contents; choose a new path')
    args.manifest.parent.mkdir(parents=True,exist_ok=True)
    args.manifest.write_bytes(payload)
    ledger, exclusions = [], []
    for row in records:
        status = 'included' if 'split' in row else 'excluded'
        ledger.append({**row,'status':status})
        if status == 'excluded':
            exclusions.append({**row,'status':'excluded','reason':row['exclusion_reason']})
    missing = []
    for (split,filename), original in sorted(official.items()):
        if (split,filename) not in matched_keys:
            missing.append({'id':official_id(split,original),'official_split':split,'official':original,
                            'reason':'no_confirmed_community_image','status':'excluded'})
    exclusions.extend(missing)
    duplicate_rows = [{'duplicate_group':key,'cross_original_split':len({r['original_split'] for r in members})>1,
                       'members':[{'id':r.get('id'),'original_split':r['original_split'],'official_filename':r.get('official',{}).get('filename'),
                                   'community':r['community'],'sha256':r['sha256'],'rgb_sha256':r['rgb_sha256'],
                                   'final_split':r.get('split'),'reason':r.get('exclusion_reason')} for r in members]}
                      for key,members in groups if len(members)>1]
    write_jsonl(out/'mapping_evidence.jsonl',sorted(ledger,key=lambda r:(r['original_split'],r['community']['cat'],r['community']['image_path'])))
    write_jsonl(out/'exclusions.jsonl',exclusions)
    write_json(out/'duplicate_groups.json',duplicate_rows)
    write_json(out/'stratification.json',strata)
    write_json(out/'decode_warnings.json',warning_rows)
    write_jsonl(out/'image_inventory.jsonl',[image_inventory[k] for k in sorted(image_inventory)])
    write_json(out/'source_lock.json',lock)
    summary = {'format':VERSION,'source_lock_sha256':file_hash(args.source_lock),'source_inputs':inputs,
               'official_records':dict(Counter(s for s,_ in official)),'community_records':dict(Counter(r['original_split'] for r in records)),
               'mapping_rules':dict(Counter(r['mapping_rule'] for r in records)),'unresolved_community_records':sum(not r['matched'] for r in records),
               'missing_official_records':len(missing),'exclusion_reasons':dict(Counter(r['reason'] for r in exclusions)),
               'duplicate_groups':len(duplicate_rows),'cross_original_split_duplicate_groups':sum(g['cross_original_split'] for g in duplicate_rows),
               'seed':args.seed,'split_algorithm':'sorted primary official Train category; sorted duplicate_group; Python Random(seed) shuffle in category order; round-half-up 10% groups, clamp 1..n-1 when n>1',
               'splits':{s:{'captions':sum(r['split']==s for r in manifest),'images':len({r['sha256'] for r in manifest if r['split']==s}),
                            'duplicate_groups':len({r['duplicate_group'] for r in manifest if r['split']==s})} for s in ('train','val','test')},
               'manifest_sha256':digest(payload),'exported_unique_images':len(image_inventory),'exported_image_bytes':sum(v['bytes'] for v in image_inventory.values()),
               'versions':{k:importlib.metadata.version(k) for k in ('pyarrow','Pillow','Unidecode')},'python':sys.version.split()[0],
               'builder_sha256':file_hash(Path(__file__))}
    write_json(out/'build_summary.json',summary)
    # Inference manifests contain only RGB identity/path, never references.
    for split in ('train','val','test'):
        write_jsonl(out/(split+'_rgb.jsonl'),[{'id':r['id'],'image':r['image']} for r in manifest if r['split']==split])
    print(json.dumps(summary,ensure_ascii=True),flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=Path('../data'))
    parser.add_argument('--out',type=Path,default=Path('../outputs/food500_manifest_build'))
    parser.add_argument('--manifest',type=Path,default=Path('../data/caption_manifest.jsonl'))
    parser.add_argument('--source-lock',type=Path,default=LOCK)
    parser.add_argument('--seed',type=int,choices=[42],default=42)
    parser.add_argument('--deps-dir',type=Path,help='Optional isolated dependencies built for this interpreter')
    args = parser.parse_args()
    if args.deps_dir:
        sys.path.insert(0,str(args.deps_dir.resolve()))
    build(args)


if __name__ == '__main__':
    main()
