import msgpack
import numpy as np

__all__ = ["DatumParserE2E"]


class DatumParserE2E(object):
    """This class supports serialization and deserialization of variables.

    in E2E dynamic task to read and write LMDB file.
    """

    @classmethod
    def parse_from_string(cls, raw_data) -> tuple:
        """Parse binary-bin data to e2e annos.

        Returns:
            targets: annotations for e2e dynamic perception tasks(detection
                    tracking, state estimation...).
                    shape: (len(timestamp), num_annos_each_timestamp),
                    num_annos_each_timestamp is usually equals to 15.
            targets_num_list: list of num of annos of each timestamp,
                    used to split the targets, shape: (len(timestamp), )
            target_timestamp_list: list of timestamp, shape (len(timestamp),)
            ego_raws: ego car information for e2e dynamic tasks.
                    shape: ((len(timestamp) + extra_ego_car_info),
                    num_annos_each_timestamp), extra_ego_car_info and
                    num_annos_each_timestamp, are usually equal to 11 and 7,
                    resepectively.
            target_raws: annotations for e2e dynamic trajectory prediction.
                    shape: (len(timestamp), num_annos_each_timestamp),
                    num_annos_each_timestamp is usually equals to 13.
        """
        # NOTE: all the above shape is generate from e2e data pipelien,
        # for more detail please refer to:
        # https://gitlab.hobot.cc/ptd/experimental/alg/Honeybadger/3dvision/autodatapipelinebev/-/blob/master/e2e_dynamic_tools/data_pack/pack_rec_data/step1_generate_motr_clip_list.py#L799  # noqa [E501]

        raw_data = msgpack.unpackb(raw_data, raw=False)

        start_idx = 0
        bit_len_num_list = np.frombuffer(
            raw_data[start_idx : start_idx + 4], np.uint32
        )[0]
        start_idx = start_idx + 4
        targets_num_list = np.frombuffer(
            raw_data[start_idx : start_idx + bit_len_num_list], dtype=np.uint16
        )
        start_idx = start_idx + bit_len_num_list

        bit_len_timestamp_list = np.frombuffer(
            raw_data[start_idx : start_idx + 4], np.uint32
        )[0]
        start_idx = start_idx + 4
        target_timestamp_list = np.frombuffer(
            raw_data[start_idx : start_idx + bit_len_timestamp_list],
            dtype=np.uint32,
        )
        start_idx = start_idx + bit_len_timestamp_list

        bit_len_targets = np.frombuffer(
            raw_data[start_idx : start_idx + 4], np.uint32
        )[0]
        start_idx = start_idx + 4
        targets = np.frombuffer(
            raw_data[start_idx : start_idx + bit_len_targets], dtype=np.float32
        )
        start_idx = start_idx + bit_len_targets

        bit_len_ego_raws = np.frombuffer(
            raw_data[start_idx : start_idx + 4], np.uint32
        )[0]
        start_idx = start_idx + 4
        ego_raws = np.frombuffer(
            raw_data[start_idx : start_idx + bit_len_ego_raws],
            dtype=np.float32,
        )
        start_idx = start_idx + bit_len_ego_raws

        ego_raws_column_num = np.frombuffer(
            raw_data[start_idx : start_idx + 4], np.uint32
        )[0]
        start_idx = start_idx + 4

        bit_len_target_raws = np.frombuffer(
            raw_data[start_idx : start_idx + 4], np.uint32
        )[0]
        start_idx = start_idx + 4
        target_raws = np.frombuffer(
            raw_data[start_idx : start_idx + bit_len_target_raws],
            dtype=np.float32,
        )
        start_idx = start_idx + bit_len_target_raws

        targets = targets.reshape(np.sum(targets_num_list), -1)
        target_raws = target_raws.reshape(np.sum(targets_num_list), -1)
        ego_raws = ego_raws.reshape(-1, ego_raws_column_num)
        # ego_raws = ego_raws.reshape(targets_num_list.shape[0], -1)
        return (
            targets,  # shape(len(timestamp)xnum_annos_each_timestamp, 15)
            targets_num_list,  # shape(len(timestamp),)
            target_timestamp_list,  # shape(len(timestamp),)
            ego_raws,
            target_raws,  # shape(len(timestamp)xnum_annos_each_timestamp, 13)
        )

    @classmethod
    def serialize_to_string(
        cls,
        targets,
        targets_num_list,
        target_timestamp_list,
        ego_raws,
        target_raws,
    ):
        targets_num_list = targets_num_list.astype(np.uint16).tobytes()
        target_timestamp_list = target_timestamp_list.astype(
            np.uint32
        ).tobytes()
        targets = targets.astype(np.float32).tobytes()

        ego_raws_column_num = np.uint32(ego_raws.shape[-1]).tobytes()
        ego_raws = ego_raws.reshape([-1]).astype(np.float32).tobytes()
        target_raws = target_raws.astype(np.float32).tobytes()
        bit_len_num_list = np.uint32(len(targets_num_list)).tobytes()
        bit_len_timestamp_list = np.uint32(
            len(target_timestamp_list)
        ).tobytes()
        bit_len_targets = np.uint32(len(targets)).tobytes()
        bit_len_ego_raws = np.uint32(len(ego_raws)).tobytes()
        bit_len_target_raws = np.uint32(len(target_raws)).tobytes()
        data = (
            bit_len_num_list
            + targets_num_list
            + bit_len_timestamp_list
            + target_timestamp_list
            + bit_len_targets
            + targets
            + bit_len_ego_raws
            + ego_raws
            + ego_raws_column_num
            + bit_len_target_raws
            + target_raws
        )
        return msgpack.packb(data, use_bin_type=True)
