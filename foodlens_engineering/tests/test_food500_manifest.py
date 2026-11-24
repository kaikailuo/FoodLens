"""Data-builder tests; no models, training or downloads."""
from collections import defaultdict
import copy
import json
from pathlib import Path
import tempfile
import unittest
from scripts.build_food500_manifest import (assign_splits, duplicate_components,
                                            match_record, normalized_path, load_official, digest, encoded)


class MappingTests(unittest.TestCase):
    def test_known_serializations_require_identical_canonical_records(self):
        rows=[{'caption':'original','filename':'rice/rice_01.jpg','cat':'rice'}]
        compact=encoded(rows)
        pretty=json.dumps(rows,indent=2).encode()
        spec={'sha256':digest(compact),'also_accepted_git_sha256':[digest(pretty)],
              'records':1,'canonical_records_sha256':digest(compact)}
        # Root outputs are ignored; never write fixtures into original data.
        root=Path(__file__).resolve().parents[2]/'outputs/food500_manifest_build'
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as tmp:
            path=Path(tmp)/'official.json'
            for raw in (compact,pretty):
                path.write_bytes(raw)
                self.assertEqual(load_official(path,spec),rows)
            changed=encoded([{**rows[0],'caption':'changed'}])
            path.write_bytes(changed)
            spec['also_accepted_git_sha256'].append(digest(changed))
            with self.assertRaisesRegex(ValueError,'canonical content'):
                load_official(path,spec)

    def test_transliteration_requires_unique_path_and_complete_caption(self):
        original = {'cat':'Æbleflæsk','filename':'Æbleflæsk/Æbleflæsk_0001.jpg',
                    'caption':'Æbleflæsk with sliced \u200b\u200bonion。'}
        community = {'cat':'AEbleflaesk','image':{'path':'AEbleflaesk_0001.jpg'},
                     'caption':'AEbleflaesk with sliced   onion. '}
        key = ('train',normalized_path(original['filename']))
        result,rule,_ = match_record('train',community,{key:[original]})
        self.assertEqual(result,original)
        self.assertEqual(rule,'transliterated_path_full_caption_unidecode')
        # A valid alternative description still requires review, not an automatic replacement.
        altered = {**community,'caption':'A different view of the same dish.'}
        self.assertIsNone(match_record('train',altered,{key:[original]})[0])
        # A full-caption match may not resolve a nonunique path.
        self.assertIsNone(match_record('train',community,{key:[original,original.copy()]})[0])
        self.assertIsNone(match_record('test',community,{key:[original]})[0])

    def test_unsafe_paths_rejected(self):
        for name in ('../escape.jpg','/absolute.jpg','C:/absolute.jpg'):
            with self.assertRaises(ValueError):
                normalized_path(name)


class SplitTests(unittest.TestCase):
    def test_transitive_byte_and_pixel_groups_keep_test_priority(self):
        rows = [dict(sha256='a',rgb_sha256='x',original_split='train',matched=True,official={'cat':'rice'}),
                dict(sha256='a',rgb_sha256='y',original_split='train',matched=True,official={'cat':'rice'}),
                dict(sha256='b',rgb_sha256='y',original_split='test',matched=False)]
        groups = duplicate_components(rows)
        self.assertEqual(len(groups),1)
        assign_splits(groups)
        # An unresolved Test image still bars identical pixels from training.
        self.assertEqual([r.get('split') for r in rows],[None,None,None])
        self.assertEqual(rows[0]['exclusion_reason'],'test_priority_exact_duplicate')

    def test_validation_ignores_input_order_and_preserves_multi_caption(self):
        rows = [dict(sha256=f'{i:064x}',rgb_sha256=f'{i:064x}',original_split='train',matched=True,
                     official={'cat':'rice'},id=str(i)) for i in range(20)]
        rows.append({**rows[0],'id':'extra-caption'})
        first,second = copy.deepcopy(rows),list(reversed(copy.deepcopy(rows)))
        assign_splits(duplicate_components(first))
        assign_splits(duplicate_components(second))
        self.assertEqual({r['id']:r['split'] for r in first},{r['id']:r['split'] for r in second})
        self.assertEqual(first[0]['split'],first[-1]['split'])
        self.assertEqual(len({r['duplicate_group'] for r in first if r['split']=='val'}),2)


if __name__ == '__main__':
    unittest.main()
