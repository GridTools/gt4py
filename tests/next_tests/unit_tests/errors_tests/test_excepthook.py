from gt4py.next.errors import excepthook
from gt4py.next.errors import exceptions
from gt4py import eve


def test_determine_developer_mode():
    # Env var overrides python env.
    assert excepthook._determine_developer_mode(False, False) == False
    assert excepthook._determine_developer_mode(True, False) == False
    assert excepthook._determine_developer_mode(False, True) == True
    assert excepthook._determine_developer_mode(True, True) == True

    # Defaults to python env if no env var specified.
    assert excepthook._determine_developer_mode(False, None) == False
    assert excepthook._determine_developer_mode(True, None) == True


def test_format_uncaught_error():
    try:
        loc = eve.SourceLocation("/src/file.py", 1, 1)
        msg = "compile error msg"
        raise exceptions.CompilerError(loc, msg) from ValueError("value error msg")
    except exceptions.CompilerError as err:
        str_devmode = "".join(excepthook._format_uncaught_error(err, True))
        assert str_devmode.find("Source location") >= 0
        assert str_devmode.find("Traceback") >= 0
        assert str_devmode.find("cause") >= 0
        assert str_devmode.find("ValueError") >= 0
        str_usermode = "".join(excepthook._format_uncaught_error(err, False))
        assert str_usermode.find("Source location") >= 0
        assert str_usermode.find("Traceback") < 0
        assert str_usermode.find("cause") < 0
        assert str_usermode.find("ValueError") < 0