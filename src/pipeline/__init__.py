"""LION-Bench pipeline framework.

Formalizes what used to live in ``src/check/_filter_utils.py``:

    audit.py     — per-(ep, sub) lifecycle log (events → verdict resolver)

Future homes (migrated incrementally):

    config.py    — selection-yaml merging + exp resolution
    survivor.py  — survivor.yaml schema + IO
"""
