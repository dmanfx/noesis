from __future__ import annotations

import hashlib
import copy
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "DS9/scripts/rebase_asset_realization.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("native_manifest_rebase", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rebaser = _load_module()


def _old_manifest() -> bytes:
    return subprocess.run(
        ["git", "show", "HEAD:DS9/asset_manifest.yaml"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
    ).stdout


def _fixture(tmp_path: Path) -> tuple[Path, bytes, str]:
    root = tmp_path / "artifacts"
    root.mkdir(mode=0o700)
    old_raw = _old_manifest()
    payload = {
        "schema_version": 1,
        "contract": "noesis.ds9.asset_realization",
        "base_manifest": {
            "path": "DS9/asset_manifest.yaml",
            "sha256": hashlib.sha256(old_raw).hexdigest(),
        },
        "source_contracts": {
            "path": "DS9/config/engine_source_contracts.json",
            "sha256": "a" * 64,
        },
        "created_at_utc": "2026-08-12T00:00:00Z",
        "updated_at_utc": "2026-08-12T00:00:00Z",
        "artifacts": {
            "engine.example": {
                "state": "validated",
                "provenance": {"output_sha256": "b" * 64},
            }
        },
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path = root / "asset_realization.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    return root, raw, hashlib.sha256(raw).hexdigest()


def test_native_manifest_rebase_dry_run_preserves_artifacts(dirty_old, monkeypatch) -> None:
    # Keep legacy explicit-ref API coverage independent of unrelated dirty
    # native declarations in the real checkout.
    f = dirty_old
    root, raw, digest = f['root'], f['raw'], f['digest']
    monkeypatch.setattr(rebaser, "_git_manifest", lambda _ref: f['old_raw'])
    result = rebaser.rebase(
        artifact_root=root,
        old_manifest_ref="HEAD",
        expected_realization_sha256=digest,
        apply=False,
    )
    assert result["ok"] is True
    assert result["applied"] is False
    assert result["realized_artifact_count"] == 10
    assert (root / "asset_realization.json").read_bytes() == raw


def test_native_manifest_rebase_apply_is_atomic_and_evidenced(
    dirty_old, monkeypatch
) -> None:
    f = dirty_old
    root, raw, digest = f['root'], f['raw'], f['digest']
    monkeypatch.setattr(rebaser, "_git_manifest", lambda _ref: f['old_raw'])
    result = rebaser.rebase(
        artifact_root=root,
        old_manifest_ref="HEAD",
        expected_realization_sha256=digest,
        apply=True,
    )
    assert result["applied"] is True
    updated = json.loads((root / "asset_realization.json").read_text())
    before = json.loads(raw)
    assert updated["artifacts"] == before["artifacts"]
    assert updated["base_manifest"]["sha256"] == hashlib.sha256(
        f['manifest_path'].read_bytes()
    ).hexdigest()
    evidence = Path(result["evidence"])
    assert evidence.is_file()
    assert evidence.stat().st_mode & 0o777 == 0o600


def test_manifest_rebase_rejects_realized_artifact_changes(dirty_old) -> None:
    old = rebaser._mapping_from_yaml(dirty_old['old_raw'], "old")
    new = dirty_old['current']
    try:
        rebaser._validate_rebase_change(
            old,
            new,
            realized_artifact_ids={"native.example"},
        )
    except rebaser.RebaseError as exc:
        assert "realized engine authority" in str(exc)
    else:
        raise AssertionError("realized artifact mutation was accepted")


@pytest.fixture
def dirty_old(tmp_path, monkeypatch):
    """Disposable authority/outputs only; never touch the real artifact root."""
    repo = tmp_path / 'repo'
    manifest_path = repo / 'DS9/asset_manifest.yaml'
    manifest_path.parent.mkdir(parents=True)
    # Pin the existing native authority contract, never read mutable CODE
    # manifest/provenance while another owner is building a native extension.
    manifest = {'manifest_id': 'noesis-ds9-artifacts', 'schema_version': 3,
        'target': {'native_host': {'operating_system': 'ubuntu-24.04',
            'architecture': 'x86_64', 'driver_minimum': '595.58.03',
            'gstreamer': '1.24.2', 'sdk_root': '/opt/nvidia/deepstream/deepstream-9.1',
            'cuda_root': '/usr/local/cuda-13.2', 'native_root_env': 'NOESIS_DS91_NATIVE_ROOT'}},
        'runtime': {'backend': 'native_host', 'supervisor': 'DS9/scripts/run_canonical_runtime_host.py'},
        'shared_app_data': []}
    output = repo / 'DS9/native_extensions/example.so'
    output.parent.mkdir()
    output.write_bytes(b'disposable-native-output-not-a-loaded-binary')
    source = repo / 'DS9/native/example.cpp'
    source.parent.mkdir()
    source.write_text('// disposable source\n')
    native = dict(id='native.example', kind='native_extension', state='validated',
        output='DS9/native_extensions/example.so', sources=['DS9/native/example.cpp'],
        builder='DS9/scripts/example.sh', provenance={'output_sha256': '0' * 64})
    engines = [dict(id=f'engine.existing.{i}', kind='tensorrt_engine',
        state='validated', output=f'unopened-engine-{i}.engine',
        provenance={'output_sha256': str(i) * 64}) for i in range(10)]
    manifest['artifacts'] = [native] + engines
    old_raw = (yaml.safe_dump(manifest, sort_keys=False) + '# dirty old, not HEAD\n').encode()
    old_path = tmp_path / 'old-current.yaml'
    old_path.write_bytes(old_raw)
    current = copy.deepcopy(manifest)
    current['artifacts'][0]['provenance']['output_sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest_path.write_text(yaml.safe_dump(current, sort_keys=False))
    root = tmp_path / 'artifacts'
    root.mkdir(mode=0o700)
    realization = dict(schema_version=1, contract=rebaser.REALIZATION_CONTRACT,
        base_manifest={'path': 'DS9/asset_manifest.yaml', 'sha256': hashlib.sha256(old_raw).hexdigest()},
        source_contracts={'path': 'DS9/config/engine_source_contracts.json', 'sha256': 'a' * 64},
        artifacts={record['id']: copy.deepcopy(record) for record in engines},
        created_at_utc='2026-08-12T00:00:00Z', updated_at_utc='2026-08-12T00:00:00Z')
    raw = (json.dumps(realization, indent=2, sort_keys=True) + '\n').encode()
    realization_path = root / 'asset_realization.json'
    realization_path.write_bytes(raw)
    realization_path.chmod(0o600)
    monkeypatch.setattr(rebaser, 'REPO_ROOT', repo)
    monkeypatch.setattr(rebaser, 'CURRENT_MANIFEST', manifest_path)
    return dict(root=root, old_path=old_path, old_raw=old_raw, current=current,
        manifest_path=manifest_path, realization=realization, raw=raw,
        realization_path=realization_path, digest=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize('apply', [False, True])
def test_explicit_dirty_old_manifest_path_preserves_ten_engines(dirty_old, monkeypatch, apply):
    f = dirty_old
    monkeypatch.setattr(rebaser, '_git_manifest', lambda _ref: pytest.fail('path must not read Git'))
    result = rebaser.rebase(artifact_root=f['root'], old_manifest_path=f['old_path'],
        expected_realization_sha256=f['digest'], apply=apply)
    assert result['old_manifest_sha256'] == hashlib.sha256(f['old_raw']).hexdigest()
    assert result['changed_unrealized_artifact_ids'] == ['native.example']
    assert result['realized_artifact_count'] == 10
    actual = json.loads(f['realization_path'].read_bytes())
    assert actual['artifacts'] == f['realization']['artifacts']
    assert actual['source_contracts'] == f['realization']['source_contracts']
    if not apply:
        assert f['realization_path'].read_bytes() == f['raw']
    else:
        assert actual['base_manifest']['sha256'] == hashlib.sha256(f['manifest_path'].read_bytes()).hexdigest()
        evidence = Path(result['evidence'])
        assert evidence.parent.joinpath('asset_realization.before.json').read_bytes() == f['raw']
        assert evidence.stat().st_mode & 0o777 == 0o600


def _path_rebase(f, **changes):
    args = dict(artifact_root=f['root'], old_manifest_path=f['old_path'],
        expected_realization_sha256=f['digest'], apply=False)
    args.update(changes)
    return rebaser.rebase(**args)


def test_old_manifest_path_actual_cli_and_exclusive_inputs(dirty_old, capsys):
    f = dirty_old
    args = ['--artifact-root', str(f['root']), '--old-manifest-path', str(f['old_path']),
            '--expected-realization-sha256', f['digest']]
    assert rebaser.main(args) == 0
    assert json.loads(capsys.readouterr().out)['old_manifest_sha256'] == hashlib.sha256(f['old_raw']).hexdigest()
    with pytest.raises(SystemExit) as error:
        rebaser._parse_args(args + ['--old-manifest-ref', 'HEAD'])
    assert error.value.code == 2
    with pytest.raises(rebaser.RebaseError, match='mutually exclusive'):
        _path_rebase(f, old_manifest_ref='HEAD')
    assert f['realization_path'].read_bytes() == f['raw']


@pytest.mark.parametrize('ref', [None, 'HEAD~1'])
def test_old_manifest_path_preserves_default_head_and_explicit_ref(dirty_old, monkeypatch, ref):
    f = dirty_old
    calls = []
    def git_manifest(value):
        calls.append(value)
        return f['old_raw']
    monkeypatch.setattr(rebaser, '_git_manifest', git_manifest)
    _path_rebase(f, old_manifest_path=None, old_manifest_ref=ref)
    assert calls == ['HEAD' if ref is None else ref]


@pytest.mark.parametrize('value', [True, 1, {}, []])
def test_old_manifest_path_rejects_malformed_api_type(dirty_old, value):
    with pytest.raises(rebaser.RebaseError, match='filesystem path'):
        _path_rebase(dirty_old, old_manifest_path=value)


@pytest.mark.parametrize('case', ['missing', 'directory', 'symlink', 'hardlink',
    'empty', 'oversize', 'malformed_yaml', 'nonmapping', 'different_bytes', 'different_owner', 'read_drift'])
def test_old_manifest_path_unsafe_or_unbound_bytes_refuse(dirty_old, monkeypatch, case):
    f = dirty_old
    path = f['old_path']
    if case == 'missing': path = path.with_name('absent.yaml')
    elif case == 'directory': path = path.parent
    elif case == 'symlink':
        link = path.with_name('link.yaml'); link.symlink_to(path); path = link
    elif case == 'hardlink': path.with_name('hardlink.yaml').hardlink_to(path)
    elif case == 'empty': path.write_bytes(b'')
    elif case == 'oversize': monkeypatch.setattr(rebaser, 'MAX_BYTES', len(f['old_raw']) - 1)
    elif case == 'malformed_yaml': path.write_bytes(b'[')
    elif case == 'nonmapping': path.write_bytes(b'[]')
    elif case == 'different_bytes': path.write_bytes(f['old_raw'] + b'\n')
    elif case == 'different_owner':
        from types import SimpleNamespace
        real_lstat = Path.lstat
        def lstat(p):
            info = real_lstat(p)
            if p != path: return info
            fields = {name: getattr(info, name) for name in ('st_uid', 'st_mode',
                'st_nlink', 'st_size', 'st_ino', 'st_mtime_ns', 'st_ctime_ns')}
            fields['st_uid'] += 1
            return SimpleNamespace(**fields)
        monkeypatch.setattr(Path, 'lstat', lstat)
    else:
        real_read = Path.read_bytes
        def read_bytes(p):
            raw = real_read(p)
            if p == path: p.write_bytes(raw + b'# concurrent change\n')
            return raw
        monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    with pytest.raises(rebaser.RebaseError):
        _path_rebase(f, old_manifest_path=path)
    assert f['realization_path'].read_bytes() == f['raw']


@pytest.mark.parametrize('case', ['cas', 'private_mode', 'artifact_root_mode',
    'engine', 'source_list', 'output', 'state', 'builder', 'kind',
    'output_digest', 'source_missing', 'native_host'])
def test_old_manifest_path_preserves_existing_rebase_guards(dirty_old, case):
    f = dirty_old
    changes = {}
    current = f['current']
    native = current['artifacts'][0]
    if case == 'cas': changes['expected_realization_sha256'] = '0' * 64
    elif case == 'private_mode': f['realization_path'].chmod(0o644)
    elif case == 'artifact_root_mode': f['root'].chmod(0o777)
    elif case == 'engine': current['artifacts'][1]['state'] = 'changed'
    elif case == 'source_list': native['sources'] = ['DS9/native/other.cpp']
    elif case == 'output': native['output'] = 'DS9/native_extensions/other.so'
    elif case == 'state': native['state'] = 'unverified'
    elif case == 'builder': native['builder'] = 'DS9/scripts/other.sh'
    elif case == 'kind': native['kind'] = 'tensorrt_engine'
    elif case == 'output_digest': native['provenance']['output_sha256'] = 'a' * 64
    elif case == 'source_missing':
        source = f['manifest_path'].parent / 'native/example.cpp'
        source.rename(source.with_suffix('.missing'))
    else: current['target']['native_host']['cuda_root'] = '/different'
    f['manifest_path'].write_text(yaml.safe_dump(current, sort_keys=False))
    with pytest.raises(rebaser.RebaseError):
        _path_rebase(f, **changes)
    assert f['realization_path'].read_bytes() == f['raw']


def test_old_manifest_path_preserves_commit_cas(dirty_old, monkeypatch):
    f = dirty_old
    real_write = rebaser._atomic_write
    changed = f['raw'] + b'\n'
    def write(path, raw, mode):
        real_write(path, raw, mode)
        if path.name == 'rebase_evidence.json': f['realization_path'].write_bytes(changed)
    monkeypatch.setattr(rebaser, '_atomic_write', write)
    with pytest.raises(rebaser.RebaseError, match='changed before commit'):
        _path_rebase(f, apply=True)
    assert f['realization_path'].read_bytes() == changed
