# Data Archives

This folder stores split zip parts for the generated/runtime data that is too
large or too local to keep uncompressed in git.

Restore after cloning:

```bash
python scripts/archives/restore_data_archives.py
```

Rebuild the archive parts after changing local data:

```bash
python scripts/archives/package_data_archives.py
```

The restore script verifies every part against `manifest.json` before
extracting. Existing local files are skipped unless `--overwrite` is passed.
