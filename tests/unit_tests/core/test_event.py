import torch

from hat.core.event import EventStorage


def test_event_storage():
    event = EventStorage()

    img1 = torch.rand(1, 2, 3)
    event.put("viz_img", {"image1": img1})
    img2 = torch.rand(1, 2, 3)
    event.put("viz_img", {"image2": img2})

    ret = event.get("viz_img")
    ret_keys = [list(r.keys())[0] for r in ret]
    assert "image1" in ret_keys
    assert "image2" in ret_keys

    event.put("orig_img", {"image1": img1})
    event.clear_key("viz_img")
    assert "viz_img" in event.histories
    assert len(event.histories["viz_img"]) == 0
    assert "orig_img" in event.histories

    with EventStorage() as es:
        es.put("viz_img", {"image1": img1})
        es.put("viz_img", {"image2": img2})

    ret = es.get("viz_img")
    ret_keys = [list(r.keys())[0] for r in ret]
    assert "image1" in ret_keys
    assert "image2" in ret_keys

    es.put("orig_img", {"image1": img1})
    es.clear_key("viz_img")
    assert "viz_img" in es.histories
    assert len(es.histories["viz_img"]) == 0
    assert "orig_img" in es.histories
