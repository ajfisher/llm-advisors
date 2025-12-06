class ProviderError(Exception):
    """Raised when a provider CLI call fails."""

    def __init__(self, provider: str, message: str, *, returncode: int | None = None):
        self.provider = provider
        self.returncode = returncode
        super().__init__(f"[{provider}] {message}")

