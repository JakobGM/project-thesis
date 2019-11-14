import numpy as np

from remsen.cache import Store


def test_store(tmp_path):
    """Test Store key-value cache."""
    tmp_path = tmp_path.with_suffix(".pkl")
    store = Store(path=tmp_path)
    store.insert("evaluation", "model_name", value={i: i**2 for i in range(100)})
    store.insert("training", "model_name", value=np.arange(0, 100))
    store.insert("one", "two", "three", "four", value="five")

    assert store["evaluation"]["model_name"] == {i: i**2 for i in range(100)}
    assert (store["training"]["model_name"] == np.arange(0, 100)).all()
    assert store["one"]["two"]["three"]["four"] == "five"

    del store
    store = Store(path=tmp_path)
    assert store["evaluation"]["model_name"] == {i: i**2 for i in range(100)}
    assert (store["training"]["model_name"] == np.arange(0, 100)).all()
    assert store["one"]["two"]["three"]["four"] == "five"
