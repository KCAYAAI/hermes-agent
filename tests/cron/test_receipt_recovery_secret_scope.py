"""KC 2026-10-22: canonical receipt recovery (reconcile_pending) composed and delivered the
recovered result with NO profile secret scope. On the multiplexed gateway every scoped read in
that tail (the job's home channel, e.g. SLACK_HOME_CHANNEL) raised UnscopedSecretError, so the
receipt was retried forever ("Cron receipt recovery deferred"). Recovery must run its tail under
the job's own profile scope, exactly as run_one_job does."""

from contextlib import asynccontextmanager
import json

from cron import executions, jobs, scheduler, scheduler_authority
from hermes_cli import gateway_client


def _peer(monkeypatch, job):
    class Peer:
        async def rpc(self, method, **params):
            return {'status': 'terminal', 'job': job, 'result': [True, 'document', 'answer', None]}

    @asynccontextmanager
    async def connected():
        yield Peer()
    monkeypatch.setattr(gateway_client, 'connect_gateway', connected)


def test_recovery_tail_runs_under_the_profile_secret_scope(tmp_path, monkeypatch):
    from agent import secret_scope
    seen = []
    real = scheduler._save_compose_deliver

    def spy(*args, **kwargs):
        # The delivery tail reads the home channel through get_secret; multiplexed with no scope
        # this raises UnscopedSecretError.
        seen.append(secret_scope.get_secret('SLACK_HOME_CHANNEL', ''))
        return real(*args, **kwargs)

    monkeypatch.setattr(scheduler, '_save_compose_deliver', spy)
    monkeypatch.setattr(secret_scope, '_MULTIPLEX_ACTIVE', True)
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')
        execution = executions.create_execution(job['id'], source='test')
        executions.mark_execution_running(execution['id'])
        with executions._transaction() as conn:
            conn.execute('UPDATE executions SET process_id=?,pid=? WHERE id=?',
                         ('old-owner', 999999, execution['id']))
        params = {'job_id': job['id'], 'request_id': execution['id'], 'extra_prompt': None}
        journal = scheduler_authority.journal_path(job['id'], execution['id'])
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(json.dumps({'params': params, 'receipt': None}))
        _peer(monkeypatch, job)
        scheduler_authority.reconcile_pending()
        assert seen == ['']  # read through a bound scope, no UnscopedSecretError
        assert executions.get_execution(execution['id'])['status'] == 'completed'
        assert not journal.exists()
    assert secret_scope._bound_secrets() is None  # the scope does not outlive the tail


def test_recovery_reads_the_job_profiles_value_not_the_launch_profiles(tmp_path, monkeypatch):
    """Review: an empty value cannot tell the right scope from another one. Give the launch and
    job profiles distinct home channels and require the job profile's."""
    from agent import secret_scope
    from cron import scheduler as sched
    from tui_gateway import launch_profile_policy
    seen = []
    real = scheduler._save_compose_deliver

    def spy(*args, **kwargs):
        seen.append(secret_scope.get_secret('SLACK_HOME_CHANNEL', ''))
        return real(*args, **kwargs)

    job_home = tmp_path / 'job-home'
    job_home.mkdir()
    monkeypatch.setattr(scheduler, '_save_compose_deliver', spy)
    monkeypatch.setattr(secret_scope, '_MULTIPLEX_ACTIVE', True)
    monkeypatch.setenv('SLACK_HOME_CHANNEL', 'C_LAUNCH')
    monkeypatch.setattr(sched, '_get_hermes_home', lambda: str(job_home))
    monkeypatch.setattr(launch_profile_policy, 'owner_secret_scope',
                        lambda home: {'SLACK_HOME_CHANNEL': 'C_JOB'} if str(home) == str(job_home) else {})
    with jobs.use_cron_store(tmp_path / 'cron'):
        job = jobs.create_job(prompt='report', schedule='every 1h', deliver='local')
        execution = executions.create_execution(job['id'], source='test')
        executions.mark_execution_running(execution['id'])
        with executions._transaction() as conn:
            conn.execute('UPDATE executions SET process_id=?,pid=? WHERE id=?',
                         ('old-owner', 999999, execution['id']))
        params = {'job_id': job['id'], 'request_id': execution['id'], 'extra_prompt': None}
        journal = scheduler_authority.journal_path(job['id'], execution['id'])
        journal.parent.mkdir(parents=True, exist_ok=True)
        journal.write_text(json.dumps({'params': params, 'receipt': None}))
        _peer(monkeypatch, job)
        scheduler_authority.reconcile_pending()
    assert seen == ['C_JOB']


def test_failed_scope_installation_leaves_nothing_bound(monkeypatch, tmp_path):
    """Review: if installation fails partway, no scope or multiplex context may stay bound on the
    long-lived ticker thread."""
    import pytest
    from agent import secret_scope
    from cron import scheduler as sched
    monkeypatch.setattr(sched, '_get_hermes_home', lambda: str(tmp_path))
    before = (secret_scope._bound_secrets(), secret_scope._MULTIPLEX_CONTEXT.get())

    import cron.scheduler_provider as provider
    monkeypatch.setattr(provider, 'routed_profile_fire', lambda home=None: True)

    def boom(_active):
        raise RuntimeError('context failure')
    monkeypatch.setattr(secret_scope, 'set_multiplex_context', boom)
    with pytest.raises(RuntimeError):
        sched._install_fire_secret_scope()
    assert (secret_scope._bound_secrets(), secret_scope._MULTIPLEX_CONTEXT.get()) == before

    def routing_boom(home=None):
        raise RuntimeError('routing failure')
    monkeypatch.setattr(provider, 'routed_profile_fire', routing_boom)
    with pytest.raises(RuntimeError):
        sched._install_fire_secret_scope()
    assert (secret_scope._bound_secrets(), secret_scope._MULTIPLEX_CONTEXT.get()) == before


def test_body_failure_restores_nested_scope_and_context(monkeypatch, tmp_path):
    import pytest
    from agent import secret_scope
    from cron import scheduler as sched
    import cron.scheduler_provider as provider
    monkeypatch.setattr(sched, '_get_hermes_home', lambda: str(tmp_path))
    monkeypatch.setattr(provider, 'routed_profile_fire', lambda home=None: True)
    outer = secret_scope.set_secret_scope({'X': 'outer'})
    try:
        with pytest.raises(ValueError):
            with sched._fire_secret_scope():
                assert secret_scope._MULTIPLEX_CONTEXT.get() is True
                raise ValueError('body')
        assert secret_scope._bound_secrets().mapping == {'X': 'outer'}
        assert secret_scope._MULTIPLEX_CONTEXT.get() is False
    finally:
        secret_scope.reset_secret_scope(outer)
