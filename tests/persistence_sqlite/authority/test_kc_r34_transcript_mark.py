"""KC rollback window: a claim stores its transcript mark in KC r34's column when the store has it.

KC release r34 added a nullable ``session_admissions.transcript_mark`` column and wrote the claim's
mark there; its restart recovery reads it. Upstream computes the mark only on the returned claim. If
the fleet rolls back to r34, a row claimed under this release must carry its real mark, not NULL.
"""
from persistence.sqlite.authority._admission_store import admit_session_input, claim_session_input
from persistence.sqlite.authority._runtime_ends import begin_runtime_epoch
from persistence.sqlite.conversation._store.sessions import create_session


def _stored_mark(holder, admission_id):
    return holder._execute_write(lambda conn: conn.execute(
        'SELECT transcript_mark FROM session_admissions WHERE admission_id=?', (admission_id,)).fetchone()[0])


def _admit_and_claim(holder):
    create_session(holder, 's', 'test')
    epoch = begin_runtime_epoch(holder, instance_id='boot')
    accepted, _ = admit_session_input(holder, epoch=epoch, principal_id='human', session_id='s',
                                      request_id='input', payload={'text': 'work'})
    return accepted, claim_session_input(holder, epoch=epoch, session_id='s')


def test_a_claim_on_an_r34_store_writes_its_mark_to_the_r34_column(backend, scope_a):
    holder = backend.open(scope_a).holder()
    holder._execute_write(lambda conn: conn.execute(
        'ALTER TABLE session_admissions ADD COLUMN transcript_mark INTEGER'))
    accepted, claim = _admit_and_claim(holder)
    assert claim['status'] == 'started'
    stored = _stored_mark(holder, accepted['admission_id'])
    assert stored is not None
    assert stored == claim['transcript_mark']


def test_a_claim_on_a_store_without_the_r34_column_is_unchanged(backend, scope_a):
    holder = backend.open(scope_a).holder()
    accepted, claim = _admit_and_claim(holder)
    assert claim['status'] == 'started' and isinstance(claim['transcript_mark'], int)
    cols = holder._execute_write(lambda conn: [c[1] for c in conn.execute('PRAGMA table_info(session_admissions)')])
    assert 'transcript_mark' not in cols
