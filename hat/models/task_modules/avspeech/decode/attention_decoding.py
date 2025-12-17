# Copyright (c) Horizon Robotics, All rights reserved.
import torch
from torch import nn


class Att_Decoding(nn.Module):
    """Attention decoding.

    Args:
        beam_size (int): beam size for beam search.
        vocab_path (str): path to vocab file.
    """

    def __init__(self, beam_size: int, vocab_path: str):
        super().__init__()
        self.beam_size = beam_size
        self.vocab_path = vocab_path
        self.read_symbol_table()

    def read_symbol_table(self):
        symbol_table = {}
        with open(self.vocab_path, "r", encoding="utf8") as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                symbol_table[arr[0]] = int(arr[1])
        return symbol_table

    def subsequent_mask(
        self,
        size: int,
        device,
    ) -> torch.Tensor:
        """Create mask for subsequent steps (size, size).

        This mask is used only in decoder which works in an auto-regressive
        mode.
        This means the current step could only do attention with its left
        steps.

        In encoder, fully attention is used when streaming is not necessary and
        the sequence is not long. In this  case, no attention mask is needed.

        When streaming is need, chunk-based attention is used in encoder. See
        subsequent_chunk_mask for the chunk-based attention mask.

        Args:
            size (int): size of mask
            str device (str): "cpu" or "cuda" or torch.Tensor.device
            dtype (torch.device): result dtype

        Returns:
            torch.Tensor: mask

        Examples:
            >>>> subsequent_mask(3)
            [[1, 0, 0],
            [1, 1, 0],
            [1, 1, 1]]
        """
        arange = torch.arange(size, device=device)
        mask = arange.expand(size, size)
        arange = arange.unsqueeze(-1)
        mask = mask <= arange
        return mask

    def mask_finished_scores(
        self, score: torch.Tensor, flag: torch.Tensor
    ) -> torch.Tensor:
        """Mask finished scores.

        If a sequence is finished, we only allow one alive branch. This
        function aims to give one branch a zero score and the rest -inf score.

        Args:
            score (torch.Tensor): A real value array with shape
                (batch_size * beam_size, beam_size).
            flag (torch.Tensor): A bool array with shape
                (batch_size * beam_size, 1).

        Returns:
            torch.Tensor: (batch_size * beam_size, beam_size).
        """
        # mask_finished_scores(top_k_logp, end_flag)
        beam_size = score.size(-1)
        zero_mask = torch.zeros_like(flag, dtype=torch.bool)  # (B*N, 1)
        if beam_size > 1:
            unfinished = torch.cat(
                (zero_mask, flag.repeat([1, beam_size - 1])), dim=1
            )
            finished = torch.cat(
                (flag, zero_mask.repeat([1, beam_size - 1])), dim=1
            )
        else:
            unfinished = zero_mask
            finished = flag
        score.masked_fill_(unfinished, -float("inf"))
        score.masked_fill_(finished, 0)
        return score

    def mask_finished_preds(
        self, pred: torch.Tensor, flag: torch.Tensor, eos: int
    ) -> torch.Tensor:
        """

        If a sequence is finished, all of its branch should be <eos>.

        Args:
            pred (torch.Tensor): A int array with shape
                (batch_size * beam_size, beam_size).
            flag (torch.Tensor): A bool array with shape
                (batch_size * beam_size, 1).

        Returns:
            torch.Tensor: (batch_size * beam_size).
        """
        beam_size = pred.size(-1)
        finished = flag.repeat([1, beam_size])
        return pred.masked_fill_(finished, eos)

    def forward(
        self,
        encoder_out,
        sos,
        eos,
        att_decoder,
        att_decoder_embed,
        att_decoder_mask,
        ys_lens,
    ):
        # 修改label的标签
        # ys_in       2, 256, 1, 10  hyps_pad    2(beam) 256  1 17
        # ys_mask     2, 1, 10, 10   ys_mask     2 1 7 7
        # encoder_out 2, 256, 1, 47  encoder out 2 44 256
        # mask_pad    2, 1, 1, 47    encoder mask2 1 44
        # att_out     2, 10, 4233
        """Apply beam search on attention decoder.

        Args:
            speech (torch.Tensor): (batch, max_len, feat_dim)
            speech_length (torch.Tensor): (batch, )
            beam_size (int): beam size for beam search
            decoding_chunk_size (int): decoding chunk for dynamic chunk
                trained model.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
                0: used for training, it's prohibited here
            simulate_streaming (bool): whether do encoder forward in a
                streaming fashion

        Returns:
            torch.Tensor: decoding result, (batch, max_result_len)
        """
        beam_size = self.beam_size
        batch_size = encoder_out.size(0)  # B, 256, 1, 102
        device = encoder_out.device
        maxlen = encoder_out.size(3)
        maxlen1 = ys_lens.item()
        running_size = batch_size * beam_size
        encoder_mask = torch.ones(
            beam_size, 1, 1, maxlen, dtype=torch.bool, device=device
        )  # 10, 1, 1, 102

        hyps = torch.ones(
            [running_size, 1], dtype=torch.long, device=device  # 预测字符
        ).fill_(
            sos
        )  # (B*N, 1) 10,1
        # hyps_lens = torch.tensor(
        #     [hyp.size(0) for hyp in hyps], device=device, dtype=torch.long
        # )
        scores = torch.tensor(
            [0.0] + [-float("inf")] * (beam_size - 1), dtype=torch.float
        )  # N
        scores = (
            scores.to(device).repeat([batch_size]).unsqueeze(1).to(device)
        )  # (B*N, 1)  #得分[0 ，-inf,-inf...]
        end_flag = torch.zeros_like(
            scores, dtype=torch.bool, device=device
        )  # [false,f,f,f,...]
        # cache: Optional[List[torch.Tensor]] = None
        encoder_out = encoder_out.repeat(self.beam_size, 1, 1, 1)
        # 2. Decoder forward step by step
        for i in range(1, maxlen1 + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            # encoder_out = encoder_out.repeat(self.beam_size, 1, 1, 1)
            hyps1, _ = att_decoder_embed(hyps)  # 2, 7, 256  10,1,256
            hyps1 = (
                hyps1.unsqueeze(0).permute((1, 3, 0, 2)).contiguous()
            )  # 10, 256, 1, 1
            hyps_mask = (
                self.subsequent_mask(i, device)
                .unsqueeze(0)
                .repeat(running_size, 1, 1)
                .to(device)
            )  # (B*N, i, i)  #字符mask
            hyps_mask = hyps_mask.unsqueeze(1)
            # hyps_mask, _ = att_decoder_mask(hyps_lens, hyps1.size(-1))

            # logp: (B*N, vocab)
            # logp, cache = self.decoder.forward_one_step(
            #     encoder_out, encoder_mask, hyps, hyps_mask, cache)
            # ##logp 每次自解码结果
            # hyps 10,1                   b, 256, 1, 1    b, 256, 1, 10
            # hyps_mask  10 1 1           b, 1, 1, 1      b, 1, 10, 10
            # encoder_out 10, 102, 256    b, 256, 1, 93  b, 256, 1, 47
            # encoder_mask 10, 1, 1, 102  b, 1, 1, 93    b, 1, 1, 47
            logp = att_decoder(
                hyps1, hyps_mask, encoder_out, encoder_mask
            )  # B 4233 1 1
            # logp [4233]
            decoder_out = (
                logp.squeeze(2).permute((0, 2, 1)).contiguous()
            )  # 10, 1, 4233
            decoder_out = decoder_out[:, -1, :].log_softmax(dim=-1)  #
            # decoder_out = torch.nn.functional.log_softmax(
            #     decoder_out, dim=-1)#$10 1 4233
            # # decoder_out=decoder_out.squeeze(1)#10, 4233
            # decoder_out = decoder_out[:,-1,:]
            # decoder_out[:,-1,:]
            # decoder_out = decoder_out[-1]
            # 2.2 First beam prune: select topk best prob at current time
            top_k_logp, top_k_index = decoder_out.topk(
                beam_size
            )  # (B*N, N) 分值
            top_k_logp = self.mask_finished_scores(
                top_k_logp, end_flag
            )  # (B*N) 更新score，0，-inf
            top_k_index = self.mask_finished_preds(
                top_k_index, end_flag, eos
            )  # 更新eos
            # 2.3 Second beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add  分值相加
            scores = scores.view(
                batch_size, beam_size * beam_size
            )  # (B, N*N) B,10*10  100个预测中找10个
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)
            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = (
                torch.arange(batch_size, device=device)
                .view(-1, 1)
                .repeat([1, beam_size])
            )  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size  # (B, N),?
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1
            )  # (B*N)
            # 2.5 Update best hyps
            best_k_pred = torch.index_select(
                top_k_index.view(-1),  # 根据bestk 在topk中检索，找到idx
                dim=-1,
                index=best_k_index,
            )  # (B*N)
            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index
            )  # (B*N, i) 找每层字符
            hyps = torch.cat(
                (last_best_k_hyps, best_k_pred.view(-1, 1)), dim=1
            )  # (B*N, i+1) 最终预测的字符

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], eos).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = (
            best_index
            + torch.arange(batch_size, dtype=torch.long, device=device)
            * beam_size
        )
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)
        best_hyps = best_hyps[:, 1:]  # [1, 14]
        best_hyps = best_hyps.cpu().numpy().tolist()
        # aa =hyps[best_index][0]
        aa = best_hyps[0]
        symbol_table = self.read_symbol_table()
        # Load dict
        char_dict = {v: k for k, v in symbol_table.items()}
        eos = len(char_dict) - 1
        content = ""
        for w in aa:
            if w == eos:
                break
            content += char_dict[w]
            content += " "
        att_rescroing_out = content[0:-1]
        return [att_rescroing_out]
        # return best_hyps, best_scores
