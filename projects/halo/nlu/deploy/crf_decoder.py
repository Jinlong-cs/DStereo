"""CRF解码代码."""


import os
import struct

import numpy as np


class CRF:
    """CRF decoder for nlu slots parsing."""

    def __init__(self, crf_transition_path, batch_first=False):
        """Init CRF class.

        Args:
            crf_transition_path(str): Path to crf transitions bin file.
            batch_first (bool): Whether to put batch_size at firts dimension.

        """
        self.batch_first = batch_first
        self.start_transitions = self._load_crf_start_transitions(
            crf_transition_path
        )
        self.end_transitions = self._load_crf_end_transitions(
            crf_transition_path
        )
        self.transitions = self._load_crf_transitions(crf_transition_path)
        self.num_tags = self.start_transitions.size
        print(self.start_transitions)
        print(self.end_transitions)
        print(self.transitions)
        print(self.num_tags)

    def _load_crf_start_transitions(self, crf_transition_path):
        filepath = os.path.join(
            crf_transition_path, "crf.start_transitions.bin"
        )
        start_transitions = []
        with open(filepath, "rb") as f:
            start_transitions_shape = struct.unpack("i", f.read(4))
            print(
                "start_transitions_shape: {}".format(start_transitions_shape)
            )
            for _ in range(start_transitions_shape[0]):
                start_transitions.append(struct.unpack("f", f.read(4))[0])
        print("Successfully load crf start transitions!")
        return np.array(start_transitions)

    def _load_crf_end_transitions(self, crf_transition_path):
        filepath = os.path.join(crf_transition_path, "crf.end_transitions.bin")
        end_transitions = []
        with open(filepath, "rb") as f:
            end_transitions_shape = struct.unpack("i", f.read(4))
            print("end_transitions_shape: {}".format(end_transitions_shape))
            for _ in range(end_transitions_shape[0]):
                end_transitions.append(struct.unpack("f", f.read(4))[0])
        print("Successfully load crf end transitions!")
        return np.array(end_transitions)

    def _load_crf_transitions(self, crf_transition_path):
        filepath = os.path.join(crf_transition_path, "crf.transitions.bin")
        transitions = []
        with open(filepath, "rb") as f:
            crf_transitions_shape = struct.unpack("2i", f.read(8))
            print("transitions_shape: {}".format(crf_transitions_shape))
            for i in range(crf_transitions_shape[0]):
                transitions.append([])
                for _ in range(crf_transitions_shape[1]):
                    transitions[i].append(struct.unpack("f", f.read(4))[0])
        print("Successfully load transitions!")
        return np.array(transitions)

    def decode(self, emissions, mask=None):
        """CRF decode procedure.

        Args:
            emissions (numpy.ndarray): Slots outputs logits.
            mask (numpy.ndarray): Decode mask for valid word.

        Returns:
            list: List of slots tag ids.

        """
        # emissions：[batch_size, seq_length, num_tags]
        # mask: [batch_size, seq_length]
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = np.ones(emissions.shape[:2], dtype=np.uint8)
        # print("emissions shape: {}".format(emissions.shape))
        # print("mask shape: {}".format(mask.shape))

        if self.batch_first:
            emissions = emissions.transpose((1, 0, 2))
            mask = mask.transpose((1, 0))

        return self._viterbi_decode(emissions, mask)

    def _validate(self, emissions, mask):
        if emissions.ndim != 3:
            raise ValueError(
                f"emissions must have dimension of 3, got {emissions.dim()}"
            )
        if emissions.shape[2] != self.num_tags:
            raise ValueError(
                f"expected last dimension of emissions is {self.num_tags}, "
                f"got {emissions.shape[2]}"
            )
        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    "the first two dimensions of emissions "
                    f"and mask must match, got "
                    f"{tuple(emissions.shape[:2])} and {tuple(mask.shape)}"
                )
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError("mask of the first timestep must all be on")

    def _viterbi_decode(self, emissions, mask):
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.ndim == 3 and mask.ndim == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.shape[2] == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = np.expand_dims(score, axis=2)
            broadcast_emission = np.expand_dims(emissions[i], axis=1)

            raw_next_score = (
                broadcast_score + self.transitions + broadcast_emission
            )
            # print(broadcast_score.shape)
            # print(self.transitions.shape)
            # print(broadcast_emission.shape)

            # print(np.max(raw_next_score, axis=1).shape)
            # print(np.argmax(raw_next_score, axis=1).shape)

            next_score = np.max(raw_next_score, axis=1).tolist()[0]
            indices = np.argmax(raw_next_score, axis=1).tolist()[0]

            score = np.where(
                np.expand_dims(mask[i], axis=1), next_score, score
            )
            history.append(indices)

        score += self.end_transitions

        seq_ends = mask.astype(np.long).sum(axis=0)[0] - 1
        best_tags_list = []

        for idx in range(batch_size):
            best_last_tag = np.argmax(score[idx], axis=0).tolist()
            best_tags = [best_last_tag]
            for hist in list(reversed(history[:seq_ends])):
                best_last_tag = hist[best_tags[-1]]
                best_tags.append(best_last_tag)

            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list
