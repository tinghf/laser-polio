import subprocess


def test_main():
    assert subprocess.check_output(["laser-polio", "foo", "foobar"], text=True) == "foobar\n"
