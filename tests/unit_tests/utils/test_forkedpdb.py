import pytest
import torch.multiprocessing as mp


def _worker(rank):
    if rank == 0:
        import hat.utils.forkedpdb as pdb

        pdb.set_trace()
        # import pdb; pdb.set_trace()
    print(f"i am subprocess-{rank}")


@pytest.mark.skipif(True, reason="use python3 directly")
def test_forkedpdb():
    mp.spawn(_worker, nprocs=2)


if __name__ == "__main__":
    test_forkedpdb()
