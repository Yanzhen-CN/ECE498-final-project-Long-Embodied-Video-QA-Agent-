# memory/exceptions.py
class MemoryError(Exception):
    """Base exception for memory module."""


class SchemaValidationError(MemoryError):
    """Raised when a record does not conform to schema."""


class RecordNotFound(MemoryError):
    """Raised when a record is not found."""


class DuplicateRecordError(MemoryError):
    """Raised when insert policy forbids overwriting an existing record."""


class StorageIOError(MemoryError):
    """Raised on IO/DB errors."""
