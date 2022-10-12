import nox


@nox.session
def tests(session):
    session.install("pip", "numpy", "pytest", "ase")
    session.run("pip", "install", ".", "-v")
    session.run("pytest")
