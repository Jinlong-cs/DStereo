from hat.models.base_modules.extend_container import ExtSequential
from hat.registry import build_from_registry


def test_recursive_build():
    cfg = {
        "aaa": "bbb",
        "model": {
            "type": "ExtSequential",
            "modules": [],
        },
    }

    m1 = build_from_registry(cfg)
    assert isinstance(m1["model"], ExtSequential)

    cfg.update(__build_recursive=False)

    m2 = build_from_registry(cfg)
    assert isinstance(m2["model"], dict)
