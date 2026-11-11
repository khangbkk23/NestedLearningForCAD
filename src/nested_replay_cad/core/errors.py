"""Typed failures; no partial-state or silent-zero recovery."""


class A0Error(Exception):
    exit_code = 3

    def __init__(self, reason: str, *, attempt: str | None = None, cursor: int | None = None):
        super().__init__(reason)
        self.reason, self.attempt, self.cursor = reason, attempt, cursor


class SchemaError(A0Error):
    exit_code = 2


class IdentityMismatch(A0Error):
    exit_code = 2


class IllegalDataAccess(A0Error):
    exit_code = 2


class NonFiniteState(A0Error): pass


class ParameterLayoutMismatch(A0Error): pass


class BudgetStop(A0Error):
    exit_code = 4


class ScopeBlocked(A0Error):
    exit_code = 4


class GateBlocked(A0Error):
    exit_code = 5

