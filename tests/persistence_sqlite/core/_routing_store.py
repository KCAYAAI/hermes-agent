"""KC: the SQLite backend's store under the ``SessionDB`` names the set-based routing tests call.

The tests reach the holder's writer connection, lock and write path (``_conn``, ``_lock``, ``_execute_write``,
``_read_ctx``) to count statements and inject retries; the routing rewrite runs through the store's own
``routing.replace_routing_entries``. Patch the holder (``db.holder``), never this adapter."""

from __future__ import annotations

from persistence.sqlite.conversation._store.sessions import create_session


class RoutingDB:
    def __init__(self, store):
        self.store = store

    @property
    def holder(self):
        return self.store.holder()

    def replace_gateway_routing_entries(self, entries, *, scope=""):
        return self.store.routing.replace_routing_entries(entries, routing_scope=scope)

    def create_session(self, session_id, source):
        create_session(self.holder, session_id, source)

    def close(self):
        self.holder.close()

    def __getattr__(self, name):
        return getattr(self.store.holder(), name)
