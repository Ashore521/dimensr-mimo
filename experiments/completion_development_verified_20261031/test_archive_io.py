from pathlib import Path

import pytest

from experiments.completion_development_verified_20261031 import archive_io


def test_deep_archive_copy_hash_and_idempotence(tmp_path):
    source = tmp_path/'source.txt'
    source.write_bytes(b'preserved frozen bytes')
    digest = archive_io.sha(source)
    relative = Path('nested_input')
    while len(str(tmp_path/relative/'snapshot.txt')) <= 330:
        relative /= 'frozen_inputs_long_path_segment'
    destination = tmp_path/relative/'snapshot.txt'
    archive_io.copy_original(source, destination, digest)
    archive_io.copy_original(source, destination, digest)
    assert archive_io.sha(destination) == digest
    assert archive_io.disk_path(archive_io.disk_path(destination)) == archive_io.disk_path(destination)
    archive_io.verify_files({(relative/'snapshot.txt').as_posix(): digest}, tmp_path)


def test_conflicting_archive_is_preserved_and_traversal_rejected(tmp_path):
    source, destination = tmp_path/'source.txt', tmp_path/'snapshot.txt'
    source.write_bytes(b'original')
    destination.write_bytes(b'conflicting existing evidence')
    with pytest.raises(RuntimeError, match='Existing archive drift'):
        archive_io.copy_original(source, destination, archive_io.sha(source))
    assert destination.read_bytes() == b'conflicting existing evidence'
    with pytest.raises(ValueError, match='Frozen file drift'):
        archive_io.verify_files({'../outside.txt': 'unused'}, tmp_path)
