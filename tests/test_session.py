from core.session import SignalSession


def test_empty_session():
    session = SignalSession()

    assert session.raw is None
    assert session.processed_signal is None
    assert isinstance(session.characteristics, dict)
    assert isinstance(session.spectral, dict)
    assert isinstance(session.noise, dict)
    assert isinstance(session.ml, dict)
    assert isinstance(session.diagnosis, dict)


def test_session_logging():
    session = SignalSession()

    session.add_log("Test message")

    assert len(session.log) == 1
    assert session.log[0] == "Test message"