import os

_app = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
with open(_app, encoding="utf-8") as _f:
    exec(compile(_f.read(), _app, "exec"), {"__file__": _app, "__name__": "__main__"})
