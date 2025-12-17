from hat.utils.global_var import get_value, global_dict, set_value


def test_set_and_get_value():
    set_value("name", "HAT")
    assert "name" in global_dict
    assert get_value("name") == "HAT"
    assert get_value("name_a") is None
