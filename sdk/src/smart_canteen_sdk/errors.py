class CanteenError(ValueError):
    """Stable, public error code with a human-readable message."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
