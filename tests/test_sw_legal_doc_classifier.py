import sw_legal_doc_classifier


def test_import() -> None:
    """Test that the package can be imported without errors."""
    assert isinstance(sw_legal_doc_classifier.__name__, str)
