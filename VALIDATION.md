# Package checks

Checked on 2026-10-24 with Python 3.10.19 and the versions listed in `requirements-tested.txt`.

| Entry point | Protocol/regression tests |
| --- | ---: |
| Main model | 23 passed |
| Analysis experiments | 14 passed |
| GNN comparators | 25 passed |
| Total | 62 passed |

All three data/environment checks passed. The CSV checksum matches the source dataset. The protocol tests use mock or small synthetic fixtures.

Repeat the checks with:

```bash
python check_installation.py --tests
```
