"""Pytest configuration for middleware tests."""


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "system_embedding: tests that require a real embedding API "
        "(set EMBEDDING_API_URL to enable)",
    )
    config.addinivalue_line(
        "markers",
        "e2e: end-to-end system tests requiring running ClickHouse, "
        "middleware, and ChromaDB containers",
    )
