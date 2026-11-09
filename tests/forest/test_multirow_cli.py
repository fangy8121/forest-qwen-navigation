import json
import sys
from collections import Counter

import pytest

from forest_generator import __main__ as cli
from scripts.generate_forest_multirow_dataset import records, render


def test_custom_counts_are_disjoint_and_dry_run_writes_nothing(monkeypatch,tmp_path,capsys):
    output=tmp_path/'new_batch'
    monkeypatch.setattr(sys,'argv',['forest_generator','generate-multirow','--counts','7','2','1',
        '--seed-start','400','--profiles','light','neighbor_detour','--output',str(output),'--dry-run'])
    cli.main()
    entries=json.loads(capsys.readouterr().out)
    assert [e['seed'] for e in entries]==list(range(400,410))
    assert Counter(e['split'] for e in entries)==dict(train=7,val=2,test=1)
    assert [e['profile'] for e in entries if e['split']=='val']==['light','neighbor_detour']
    assert not output.exists()


def test_explicit_seeds_choose_one_profile_without_cross_product(monkeypatch,capsys):
    monkeypatch.setattr(sys,'argv',['forest_generator','generate-multirow','--seeds','70','90',
        '--split','test','--profiles','neighbor_detour','--dry-run'])
    cli.main()
    assert json.loads(capsys.readouterr().out)==[
        dict(path='test/000',split='test',seed=70,profile='neighbor_detour'),
        dict(path='test/001',split='test',seed=90,profile='neighbor_detour')]


@pytest.mark.parametrize('extra',[
    ['--seeds','1','1'],['--counts','-1','2','1'],['--counts','0','0','0'],
    ['--seeds','1','--seed-start','3'],['--split','val'],
    ['--profiles','light','light'],['--seed-start','-1'],['--only','train/999'],
])
def test_invalid_requests_fail_before_writing(monkeypatch,tmp_path,extra):
    output=tmp_path/'invalid'
    monkeypatch.setattr(sys,'argv',['forest_generator','generate-multirow','--output',str(output),*extra])
    with pytest.raises(ValueError):cli.main()
    assert not output.exists()


def test_changed_batch_does_not_overwrite_existing_manifest(monkeypatch,tmp_path):
    original='{"maps": [], "results": []}\n'
    (tmp_path/'manifest.json').write_text(original)
    monkeypatch.setattr(sys,'argv',['forest_generator','generate-multirow','--output',str(tmp_path)])
    with pytest.raises(ValueError,match='Frozen split changed'):cli.main()
    assert (tmp_path/'manifest.json').read_text()==original
    assert not (tmp_path/'index.html').exists()


def test_gallery_uses_requested_counts(tmp_path):
    render(tmp_path,dict(maps=records(counts=(2,1,0)),results=[]))
    gallery=(tmp_path/'index.html').read_text()
    assert '已验收 0/3 张' in gallery
    assert '训练 2 / 验证 1 / 测试 0' in gallery
    assert '固定 3 个独立种子' in (tmp_path/'README.md').read_text()
