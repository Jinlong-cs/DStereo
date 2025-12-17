from hat.data.datasets.fas_dataset import FasLmdbDataset, FasRecDataset

FILENAME = ["./tmp_orig_data/face/anti_spoof/a57_posi_test.rec"]


def test_fas_rec_dataset():
    dataset = FasRecDataset(
        imgrec_path_list=FILENAME,
    )
    for idx, item in enumerate(dataset):
        assert "img" in item
        assert "fas_label" in item
        assert "database_labels" in item
        if idx > 10:
            break


def test_fas_lmdb_dataset():
    dataset = FasLmdbDataset(
        "./tmp_orig_data/face/anti_spoof/lmdb/positive/image_lmdb",
        "./tmp_orig_data/face/anti_spoof/lmdb/positive/anno_lmdb",
    )
    for idx, item in enumerate(dataset):
        assert "img" in item
        assert "fas_label" in item
        assert "car_type" in item
        if idx > 10:
            break
