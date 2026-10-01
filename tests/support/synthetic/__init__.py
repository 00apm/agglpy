"""Synthetic test cases: hand-made circle sets with hand-worked answers.

The cases are specifications, not recordings of what the code does today
(that is what the golden tests are for). Each case is plain data
(``cases.py``); an adapter runs it against an implementation and returns
a neutral ``Result`` (``result.py``). Only the adapter knows the classes
of the implementation, so the same cases check the legacy code now
(``legacy_adapter.py``) and the new core in Phase 2.
"""
