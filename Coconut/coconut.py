# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from collections import namedtuple
from transformers.models.gpt2 import GPT2LMHeadModel
import copy
Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])
MAX_N_LATENT = 8


class Coconut(nn.Module):

    def __init__(
        self,
        base_causallm,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
    ):

        super(Coconut, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id

        # tested with GPT2 and Llama3
        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):

        logits = []

        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # replace it with the preceding last hidden states
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):

        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)


class CoconutGPT_Same_Word_Embedding(nn.Module):
    def __init__(
        self,
        base_causallm,
        expainable_llm,
        # for debug
        tokenizer,
        latent_token_id,
        start_latent_id,
        end_latent_id,
        eos_token_id,
        step_start_id,
        c_thought,
        configs,
    ):

        super(CoconutGPT_Same_Word_Embedding, self).__init__()
        self.gen_forward_cnt = 0
        self.base_causallm = base_causallm
        self.expainable_llm = expainable_llm
        self.tokenizer = tokenizer
        self.latent_token_id = latent_token_id
        self.eos_token_id = eos_token_id
        self.start_latent_id = start_latent_id
        self.end_latent_id = end_latent_id
        self.step_start_id = step_start_id
        self.c_thought = c_thought
        self.config = configs

        if hasattr(self.config, "training_method"):
            if self.config.training_method == 'only_expainable_llm':
                for param in self.base_causallm.parameters():
                    param.requires_grad = False
            elif self.config.training_method == 'only_base_causallm':
                for param in self.expainable_llm.parameters():
                    param.requires_grad = False
            elif self.config.training_method == 'full':
                pass
            elif self.config.training_method == 'freeze_backbone':
                for param in self.base_causallm.parameters():
                    param.requires_grad = False
                
                for param in self.expainable_llm.parameters():
                    param.requires_grad = False
            else:
                raise ValueError(f"not this training_method {self.config.training_method=}")
            

        if isinstance(self.base_causallm, GPT2LMHeadModel):
            self.embedding = self.base_causallm.transformer.get_input_embeddings()
        else:
            self.embedding = self.base_causallm.get_input_embeddings()

    def forward(self, input_ids, attention_mask, labels, position_ids, **kwargs):
        logits = []
        loss = 0.0
        latent_indices = (
            input_ids == self.latent_token_id
        ).nonzero()  # (num_latent_tokens_in_the_batch, 2)

        latent_lists = [
            [idx[1].item() for idx in latent_indices if idx[0] == i]
            for i in range(input_ids.shape[0])
        ]  # bs, num_latent_tokens_in_the_instance (difference across the batch)

        max_n_latents = max([len(l) for l in latent_lists])

        next_compute_range = (0, input_ids.shape[1])
        inputs_embeds = self.embedding(input_ids)

        if max_n_latents > 0:
            next_compute_range = (0, latent_indices[:, 1].min().item())
            # before the earliest latent token position

        kv_cache = None

        for pass_idx in range(max_n_latents):

            if kv_cache == None:
                # first forward pass
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    output_hidden_states=True,
                )
                hidden_states_offset = 0

            else:
                # extract kv cache to reuse
                past_key_values = [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]

                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, next_compute_range[0] : next_compute_range[1], :
                    ],
                    attention_mask=attention_mask[:, : next_compute_range[1]],
                    position_ids=position_ids[
                        :, next_compute_range[0] : next_compute_range[1]
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                )

                hidden_states_offset = next_compute_range[0]
                # when we use kv_cache for the first k tokens
                # in `outputs.hidden_states`, [0, k) will be skipped
                # so we need to keep this offset to correctly use the last hidden states

            logits.append(outputs.logits)

            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if pass_idx + 1 >= max_n_latents
                    else next_compute_range[1] + 1
                ),
            )

            hidden_states = outputs.hidden_states[
                -1
            ]  # Get the last layer hidden states
            kv_cache = outputs.past_key_values

            # feedback the continuous thoughts to the input_embeds

            # first decide the positions to feedback
            filling_indices = [
                (instance_idx, mask_list[pass_idx])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > pass_idx
            ]

            # to avoid in-place operations
            # break down inputs_embeds (bs, len, hidden_size) into a list of list of 1-d tensors
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]

            # replace some of them with continuous thoughts
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair

                # replace it with the preceding last hidden states
                tensor_list[batch_idx][token_idx] = hidden_states[
                    batch_idx, token_idx - 1 - hidden_states_offset, :
                ]

            # assemble the new inputs_embeds
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # final pass
        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds[
                :, next_compute_range[0] : next_compute_range[1], :
            ],
            attention_mask=attention_mask[:, : next_compute_range[1]],
            position_ids=position_ids[:, next_compute_range[0] : next_compute_range[1]],
            past_key_values=(
                [
                    (
                        k[:, :, : next_compute_range[0], :],
                        v[:, :, : next_compute_range[0], :],
                    )
                    for k, v in kv_cache
                ]
                if kv_cache
                else None
            ),
            output_hidden_states=True,
        )

        logits.append(outputs.logits)

        self.gen_forward_cnt += max_n_latents + 1

        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss()
        if self.config.training_method == 'only_base_causallm' or self.config.training_method == 'full':
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
        
        if hasattr(self.config, 'visualize') and self.config.visualize:
            debug_predictions = []
            
            for debug_idx in range(0, len(latent_lists[0]), self.config.c_thought):
                
                continuous_embeds = inputs_embeds[:, latent_lists[0][debug_idx: debug_idx + self.c_thought], :].to(self.expainable_llm.device)
                
                if hasattr(self.config, 'w_prompt') and self.config.w_prompt:
                    if hasattr(self.config, 'explain_mode') and self.config.explain_mode == 'v1_aug':
                        thought_idx = debug_idx // 2
                        if thought_idx != 2:
                            input_explain_input_embeds_pre_order_prompt_ids = self.tokenizer(f'Step {thought_idx + 1} of the solution', add_special_tokens=False).input_ids
                        else:
                            input_explain_input_embeds_pre_order_prompt_ids = self.tokenizer(f'Step 3 and all the remaining steps of the solution', add_special_tokens=False).input_ids
                        bz = continuous_embeds.shape[0]
                        input_explain_input_embeds_pre_order_prompt_embeds = self.embedding(torch.tensor(input_explain_input_embeds_pre_order_prompt_ids).to(self.expainable_llm.device))[None, ...].repeat(bz, 1, 1)
                        continuous_embeds = torch.cat([input_explain_input_embeds_pre_order_prompt_embeds, continuous_embeds], dim=1)
                debug_ids = torch.empty((1, 0), dtype=torch.long, device=self.expainable_llm.device)
                while True:
                    if debug_ids.shape[0] != 0:
                        debug_embeds = torch.cat([continuous_embeds, self.embedding(debug_ids)], dim=1)
                    else:
                        debug_embeds = continuous_embeds
                    explainable_outputs = self.expainable_llm(
                        inputs_embeds=debug_embeds,
                        attention_mask=torch.ones(debug_embeds.shape[:2]).to(self.expainable_llm.device),
                        position_ids=torch.arange(1, debug_embeds.shape[1] + 1).unsqueeze(dim=0).to(self.expainable_llm.device),
                        output_hidden_states=True,
                    )
                    debug_logits = explainable_outputs.logits[:, -1, :] / .98
                    probs = torch.softmax(debug_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    debug_ids = torch.cat([debug_ids, next_token], dim=1)
                    print(self.tokenizer.decode(debug_ids[0]))

                    if torch.all(next_token == self.eos_token_id) or debug_ids.shape[-1] > 512: # 为 <eos>或者是'>>'结束
                        break

                debug_predictions.append(self.tokenizer.decode(debug_ids[0]))

            if hasattr(self.config, 'visualize_jsonl') and self.config.visualize_jsonl != '':
                
                save_jsonl_line(self.config.visualize_jsonl, {"predictiion": debug_predictions})    
        if hasattr(self.config, 'explain_mode') and self.config.explain_mode == 'v1_aug':
            
            if 'explainable_ids_list' in kwargs:
                c_thought_num = len(latent_lists[0]) // self.c_thought
                
                input_united_tokens = []
                def safe_token_id(x):
                    return x[0] if isinstance(x, list) else x

                start_token = safe_token_id(self.tokenizer.encode('<<', add_special_tokens=False))  # "<<"
                end_token = safe_token_id(self.tokenizer.encode('>>', add_special_tokens=False))    # ">>"
                separator_token = safe_token_id(self.tokenizer.encode('\n', add_special_tokens=False))   # "\n"
                

                def trim_trailing_zeros(group):
                    while group and group[-1] == 0:
                        group.pop()
                    return group
                
                def replace_llama_special_tokens(x, merged_token, end_token, separator_token):
                    out = []
                    for seq in x:
                        new_seq = []
                        for t in seq:
                            if t.item() == merged_token:
                                new_seq.extend([end_token, separator_token])
                            elif t.item() != 0 or len(new_seq) > 0:
                                new_seq.append(t.item())
                        out.append(torch.tensor(new_seq, device=x.device))
                    return out
                

                if len(self.tokenizer.encode('>>\n', add_special_tokens=False)) == 1:
                    merge_token = self.tokenizer.encode('>>\n', add_special_tokens=False)[0]
                    kwargs['explainable_ids_list'] = copy.deepcopy(replace_llama_special_tokens(kwargs['explainable_ids_list'], merge_token, end_token, separator_token))
                
                for j, seq in enumerate(kwargs['explainable_ids_list']):
                    i = 0
                    groups = []
                    while i < len(seq):
                        if seq[i] == start_token:
                            
                            group = [start_token]
                            i += 1
                            while i < len(seq):
                                group.append(seq[i])
                                if seq[i] == end_token:
                                    break
                                i += 1
                            group = trim_trailing_zeros(group)
                            groups.append(group)
                        else:
                            i += 1

                    if len(groups) < self.config.max_latent_stage:
                        input_ids_j = input_ids[j].tolist()

                        try:
                            start_idx = len(input_ids_j) - 1 - input_ids_j[::-1].index(self.end_latent_id)
                        except ValueError:
                            continue

                        try:
                            end_idx = input_ids_j.index(self.eos_token_id, start_idx + 1)
                        except ValueError:
                            end_idx = len(input_ids_j)

                        pseudo_thought = input_ids_j[start_idx + 1:end_idx]

                        if not pseudo_thought:
                            continue

                        if hasattr(self.config, 'format_pseudo_thought') and self.config.format_pseudo_thought:
                            tmp_num = self.tokenizer.decode(pseudo_thought).replace('### ', '')
                            pseudo_thought = self.tokenizer.encode(f'<<{tmp_num}={tmp_num}>>', add_special_tokens=False)

                        while len(groups) < c_thought_num:
                            groups.append(pseudo_thought)

                    input_united_groups = []
                    combined_group = []
                    group_count = 0

                    for group in groups:
                        group_count += 1
                        if group_count <= self.config.max_latent_stage - 1:
                            group = [-570] * self.c_thought + group + [self.eos_token_id]
                            cleaned_group = [int(x) if torch.is_tensor(x) else x for x in group]
                            input_united_groups.append(cleaned_group)
                        else:
                            if combined_group and combined_group[-1] == end_token and group[0] == start_token:
                                combined_group.append(separator_token)
                            combined_group.extend(group)

                    if combined_group:
                        final_group = [-570] * self.c_thought + combined_group + [self.eos_token_id]
                        cleaned_group = [int(x) if torch.is_tensor(x) else x for x in final_group]
                        input_united_groups.append(cleaned_group)

                    
                    input_united_tokens.append(copy.deepcopy(input_united_groups))

                ## max pad len
                bz = len(input_united_tokens)

                if hasattr(self.config, 'packing') and self.config.packing == True:
                    pass
                else:
                    for thought_idx in range(c_thought_num):
                        
                        max_pad_len = max(len(input_united_tokens[bz_idx][thought_idx]) for bz_idx in range(bz))
                        max_pad_len += 1
                        for bz_idx in range(bz):
                            token_seq = input_united_tokens[bz_idx][thought_idx]
                            pad_len = max_pad_len - len(token_seq)
                            if pad_len > 0:
                                token_seq += [self.eos_token_id] * pad_len  
                                input_united_tokens[bz_idx][thought_idx] = token_seq 


                loss_explain_all = 0.0
                
                if hasattr(self.config, 'packing') and self.config.packing == True:
                    max_pad_len = 0
                    for bz_idx in range(bz):
                        for thought_idx in range(c_thought_num):
                            continuous_embeds = inputs_embeds[bz_idx, latent_lists[bz_idx][self.c_thought * thought_idx]: latent_lists[bz_idx][self.c_thought * thought_idx + 1] + 1, :]
                            other_embeds = self.embedding(torch.tensor(input_united_tokens[bz_idx][thought_idx][self.c_thought:]).to(self.expainable_llm.device))
                            max_pad_len = max(max_pad_len, continuous_embeds.size(0) + other_embeds.size(0))  # Compute the max length once

                    input_explain_input_embeds_batch = [[] for _ in range(c_thought_num)]
                    input_explain_attention_mask_batch = [[] for _ in range(c_thought_num)]
                    input_explain_position_ids_batch = [[] for _ in range(c_thought_num)]
                    input_explain_labels_batch = [[] for _ in range(c_thought_num)]

                    # Process data
                    for thought_idx in range(c_thought_num):
                        for bz_idx in range(bz):
                            continuous_embeds = inputs_embeds[bz_idx, latent_lists[bz_idx][self.c_thought * thought_idx]: latent_lists[bz_idx][self.c_thought * thought_idx + 1] + 1, :]
                            other_embeds = self.embedding(torch.tensor(input_united_tokens[bz_idx][thought_idx][self.c_thought:]).to(self.expainable_llm.device))
                            
                            input_explain_input_embeds_batch[thought_idx].append(torch.cat([continuous_embeds, other_embeds], dim=0))

                            attention_eos_index = input_united_tokens[bz_idx][thought_idx].index(self.eos_token_id)
                            attention_explain_mask = torch.zeros(len(input_united_tokens[bz_idx][thought_idx]), dtype=int)
                            attention_explain_mask[:attention_eos_index + 1] = 1
                            input_explain_attention_mask_batch[thought_idx].append(attention_explain_mask)

                            input_explain_position_ids_batch[thought_idx].append(torch.arange(1, len(input_united_tokens[bz_idx][thought_idx]) + 1, dtype=int))

                            explain_labels = torch.tensor(input_united_tokens[bz_idx][thought_idx], dtype=int)
                            explain_labels_mask = (explain_labels != -570) & (explain_labels != self.eos_token_id)
                            explain_labels_mask[attention_eos_index] = True
                            explain_labels[~explain_labels_mask] = -100
                            input_explain_labels_batch[thought_idx].append(explain_labels)

                    # Pre-allocate padded tensors for the batch
                    input_explain_input_embeds_batch_tensor = torch.zeros(bz, c_thought_num, max_pad_len, continuous_embeds.size(-1), device=self.expainable_llm.device)
                    input_explain_attention_mask_batch_tensor = torch.zeros(bz, c_thought_num, max_pad_len, device=self.expainable_llm.device)
                    input_explain_position_ids_batch_tensor = torch.zeros(bz, c_thought_num, max_pad_len, device=self.expainable_llm.device)
                    input_explain_labels_batch_tensor = torch.full((bz, c_thought_num, max_pad_len), -100, device=self.expainable_llm.device)

                    # Concatenate and pad the tensors
                    for bz_idx in range(bz):
                        for thought_idx in range(c_thought_num):
                            input_explain_input_embeds_batch_tensor[bz_idx, thought_idx, :input_explain_input_embeds_batch[thought_idx][bz_idx].size(0)] = input_explain_input_embeds_batch[thought_idx][bz_idx]
                            input_explain_attention_mask_batch_tensor[bz_idx, thought_idx, :input_explain_attention_mask_batch[thought_idx][bz_idx].size(0)] = input_explain_attention_mask_batch[thought_idx][bz_idx]
                            input_explain_position_ids_batch_tensor[bz_idx, thought_idx, :input_explain_position_ids_batch[thought_idx][bz_idx].size(0)] = input_explain_position_ids_batch[thought_idx][bz_idx]
                            input_explain_labels_batch_tensor[bz_idx, thought_idx, :input_explain_labels_batch[thought_idx][bz_idx].size(0)] = input_explain_labels_batch[thought_idx][bz_idx]

                    # Stack the padded tensors
                    input_explain_input_embeds_batch_tensor = input_explain_input_embeds_batch_tensor.view(bz, -1, input_explain_input_embeds_batch_tensor.size(-1))
                    input_explain_attention_mask_batch_tensor = input_explain_attention_mask_batch_tensor.view(bz, -1)
                    input_explain_position_ids_batch_tensor = input_explain_position_ids_batch_tensor.view(bz, -1)
                    input_explain_labels_batch_tensor = input_explain_labels_batch_tensor.view(bz, -1)

                    # Apply 4D attention mask preparation (if necessary)
                    input_explain_attention_mask_batch_tensor = prepare_4d_attention_mask(input_explain_attention_mask_batch_tensor, dtype=self.expainable_llm.dtype)

                    # Forward pass
                    explainable_outputs = self.expainable_llm(
                        inputs_embeds=input_explain_input_embeds_batch_tensor,
                        attention_mask=input_explain_attention_mask_batch_tensor,
                        position_ids=input_explain_position_ids_batch_tensor.to(torch.long),
                        output_hidden_states=True,
                    )

                    explainable_logits = explainable_outputs.logits
                    effective_loss_num = float((input_explain_labels_batch_tensor != -100).sum(dim=1).bool().sum().item())

                    shift_explain_logits = explainable_logits[..., :-1, :].contiguous()
                    shift_explain_labels = input_explain_labels_batch_tensor[..., 1:].to(torch.long).contiguous()
                    loss_explain_fct = CrossEntropyLoss(reduction='sum')
                    loss_explain = loss_explain_fct(
                        shift_explain_logits.view(-1, shift_explain_logits.size(-1)), shift_explain_labels.view(-1)
                    )

                    loss_explain /= effective_loss_num
                    loss_explain_all += loss_explain

                else:
                    
                    for thought_idx in range(c_thought_num):
                        input_explain_input_embeds = []
                        input_explain_attention_mask, input_explain_position_ids, input_explain_labels = [], [], []
                        max_pad_len = -1

                        def extract_token_range(tensor, start_id=128000, end_id=128256):
                            try:
                                start_idx = (tensor == start_id).nonzero(as_tuple=True)[0][0].item()
                                end_idx = (tensor == end_id).nonzero(as_tuple=True)[0][0].item()
                                return tensor[start_idx:end_idx]
                            except IndexError:
                                print("start_id or end_id not in tensor")
                                return None

                        for bz_idx in range(bz):
                            
                            latent_len = len(latent_lists[bz_idx])
                            start_idx = thought_idx * self.c_thought
                            end_idx = min(start_idx + self.c_thought, latent_len)
                            continuous_embeds = inputs_embeds[bz_idx, latent_lists[bz_idx][start_idx:end_idx], :]
                            
                            other_embeds = self.embedding(torch.tensor(input_united_tokens[bz_idx][thought_idx][self.c_thought:]).to(self.expainable_llm.device))
                            input_explain_input_embeds.append(torch.cat([continuous_embeds, other_embeds], dim=0))
                            attention_eos_index = input_united_tokens[bz_idx][thought_idx].index(self.eos_token_id)
                            attention_explain_mask = torch.zeros(len(input_united_tokens[bz_idx][thought_idx]), dtype=int)
                            attention_explain_mask[:attention_eos_index+1] = 1
                            input_explain_attention_mask.append(attention_explain_mask)
                            input_explain_position_ids.append(torch.arange(1, len(input_united_tokens[bz_idx][thought_idx]) + 1, dtype=int))
                            explain_labels = torch.tensor(input_united_tokens[bz_idx][thought_idx], dtype=int)
                            explain_labels_mask = (explain_labels != -570) & (explain_labels != self.eos_token_id)
                            explain_labels_mask[attention_eos_index] = True
                            explain_labels[~explain_labels_mask] = -100
                            input_explain_labels.append(explain_labels)
                        
                        input_explain_input_embeds = torch.stack(input_explain_input_embeds)
                        input_explain_attention_mask = torch.stack(input_explain_attention_mask)
                        input_explain_position_ids = torch.stack(input_explain_position_ids)
                        input_explain_labels = torch.stack(input_explain_labels)
                        
                        
                        explainable_outputs = self.expainable_llm(
                            inputs_embeds=input_explain_input_embeds.to(self.expainable_llm.device),
                            attention_mask=input_explain_attention_mask.to(self.expainable_llm.device),
                            position_ids=input_explain_position_ids.to(self.expainable_llm.device),
                            output_hidden_states=True,
                        )
                        if hasattr(self.config, "use_prj") and self.config.use_prj:
                            explainable_logits = self.base_causallm.lm_head(self.projector2(explainable_outputs.hidden_states[-1]))
                        else:
                            explainable_logits = explainable_outputs.logits
                        
                        if hasattr(self.config, "loss_level") and self.config.loss_level == 'token_level':
                            effective_token_count = (input_explain_labels != -100).sum()
                        else:
                            effective_token_count = float((input_explain_labels != -100).sum(dim=1).bool().sum().item())

                        shift_explain_logits = explainable_logits[..., :-1, :].contiguous()
                        shift_explain_labels = input_explain_labels[..., 1:].contiguous()
                        loss_explain_fct = CrossEntropyLoss(reduction='sum')
                        loss_explain = loss_explain_fct(
                            shift_explain_logits.view(-1, shift_explain_logits.size(-1)).to(self.expainable_llm.device), shift_explain_labels.view(-1).to(self.expainable_llm.device)
                        )
                        loss_explain /= effective_token_count
                        loss_explain_all += loss_explain
                        
        if 'explainable_ids_list' in kwargs:
            if loss is None:
                loss = 0.0
            loss += 1.0 * loss_explain_all / c_thought_num

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    def train(self):
        self.base_causallm.train()

    def eval(self):
        self.base_causallm.eval()

    def generate(
        self,
        input_ids,
        attention_mask,  # attention_mask is not used
        max_new_tokens=16,
        output_embedding=False,
        synced_gpus=False,
        **kwargs
    ):
        
        self.gen_forward_cnt = 0

        assert input_ids.shape[0] == 1, "only support batch_size == 1 now"

        tokens = input_ids[0].detach().tolist()

        labels = input_ids.clone()  # placeholder. not used.
        outputs = self.forward(
            input_ids,
            torch.ones_like(input_ids, device=input_ids.device),
            labels,
            torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).reshape(1, -1),
        )
        inputs_embeds = outputs.inputs_embeds

        # get the first token using the current hidden state
        next_token = torch.argmax(outputs.logits[0, -1]).item()
        tokens.append(next_token)
        new_token_embed = self.embedding(
            torch.tensor(next_token, device=input_ids.device)
        ).view(1, 1, -1)
        new_inputs_embeds = torch.cat((inputs_embeds, new_token_embed), dim=1)

        # get other tokens
        for _ in range(max_new_tokens - 1):
            outputs = self.base_causallm(inputs_embeds=new_inputs_embeds)
            self.gen_forward_cnt += 1
            next_token = torch.argmax(outputs.logits[0, -1]).item()
            if next_token == self.eos_token_id:
                break
            tokens.append(next_token)
            new_token_embed = self.embedding(
                torch.tensor(next_token, device=input_ids.device)
            ).view(1, 1, -1)
            new_inputs_embeds = torch.cat((new_inputs_embeds, new_token_embed), dim=1)

        if synced_gpus:
            # in FSDP, the number of forward pass need to be the same across devices
            while (
                self.gen_forward_cnt < max_new_tokens + MAX_N_LATENT
            ):  # leave some room for latent tokens
                self.gen_forward_cnt += 1
                _ = self.base_causallm(inputs_embeds=new_inputs_embeds)

        if output_embedding:
            # for analysis purpose
            return torch.tensor(tokens).view(1, -1), new_inputs_embeds

        else:
            return torch.tensor(tokens).view(1, -1)