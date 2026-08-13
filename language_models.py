import openai
from openai import OpenAI
import anthropic
import os
import time
import torch
import gc
import tiktoken
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
# import google.generativeai as palm

# def get_model_path_and_template(model_name):
#     full_model_dict={
#         "minigpt":{
#             "path":"/mnt/data/data/home/wangzhiyuan/LLM-model/MiniGpt-4/alv001/MiniGpt-4-7B",
#             "template":"gpt-4"
#         },
#     }
#     # template = full_model_dict[model_name]["template"] if model_name in full_model_dict else "gpt-4"
#     assert model_name in full_model_dict, f"Model {model_name} not found in `full_model_dict` (available keys {full_model_dict.keys()})"
#     path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
#     return path, template

# def load_indiv_model(model_name, device=None):
#     model_path, template = get_model_path_and_template(model_name)
    
#     if 'gpt' in model_name or 'together' in model_name:
#         lm = GPT(model_name)
#     else:
#         model = AutoModelForCausalLM.from_pretrained(
#                 model_path, 
#                 torch_dtype=torch.float16,
#                 low_cpu_mem_usage=True, device_map="auto",
#                 token=os.getenv("HF_TOKEN"),
#                 trust_remote_code=True).eval()

#         tokenizer = AutoTokenizer.from_pretrained(
#             model_path,
#             use_fast=False,
#             token=os.getenv("HF_TOKEN")
#         )
    
#         if 'llama2' in model_path.lower():
#             tokenizer.pad_token = tokenizer.unk_token
#             tokenizer.padding_side = 'left'
#         if 'vicuna' in model_path.lower():
#             tokenizer.pad_token = tokenizer.eos_token
#             tokenizer.padding_side = 'left'
#         if 'mistral' in model_path.lower() or 'mixtral' in model_path.lower():
#             tokenizer.pad_token = tokenizer.eos_token
#             tokenizer.pad_token_id = tokenizer.eos_token_id
#         if not tokenizer.pad_token:
#             tokenizer.pad_token = tokenizer.eos_token

#         lm = HuggingFace(model_name, model, tokenizer)
    
#     return lm, template

class GPT:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 20
    API_LOGPROBS = True
    API_TOP_LOGPROBS = 20

    def __init__(self, model_name):
        self.model_name = model_name
        if 'gpt' in model_name:
            self.client = OpenAI(api_key=os.getenv("sk-gXNfRnMGRbQxr92KDfkLKZXFl1CzK5UEWZNUiEcSjZnpKfvR"))
        elif 'together' in model_name:
            TOGETHER_API_KEY = os.environ.get("TOGETHER_API_KEY")
            self.client = OpenAI(api_key=TOGETHER_API_KEY, base_url='https://api.together.xyz')
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        self.tokenizer = tiktoken.encoding_for_model("gpt-4")  # same as for gpt-3.5     
        self.tokenizer.vocab_size = 100256  # note values from 100256 to 100275 (tokenizer.max_token_value) throw an error   

    def generate(
        self, convs: List[List[Dict]], max_n_tokens: int, temperature: float, top_p: float
    ):
        """
        Args:
            convs: List of conversations (each of them is a List[Dict]), OpenAI API format
            max_n_tokens: int, max number of tokens to generate
            temperature: float, temperature for sampling
            top_p: float, top p for sampling
        Returns:
            str: generated response
        """
        outputs = []
        for conv in convs:
            output = self.API_ERROR_OUTPUT
            for _ in range(self.API_MAX_RETRY):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=conv,
                        max_tokens=max_n_tokens, 
                        temperature=temperature,
                        top_p=top_p,
                        timeout=self.API_TIMEOUT,
                        logprobs=self.API_LOGPROBS,
                        top_logprobs=self.API_TOP_LOGPROBS,
                        seed=0,
                    )
                    response_logprobs = [
                        dict((response.choices[0].logprobs.content[i_token].top_logprobs[i_top_logprob].token, 
                                response.choices[0].logprobs.content[i_token].top_logprobs[i_top_logprob].logprob) 
                                for i_top_logprob in range(self.API_TOP_LOGPROBS)
                        )
                        for i_token in range(len(response.choices[0].logprobs.content))
                    ]
                    output = {'text': response.choices[0].message.content,
                            'logprobs': response_logprobs,
                            'n_input_tokens': response.usage.prompt_tokens,
                            'n_output_tokens': response.usage.completion_tokens,
                    }
                    break
                except openai.OpenAIError as e:
                    print(type(e), e)
                    time.sleep(self.API_RETRY_SLEEP)

                time.sleep(self.API_QUERY_SLEEP)
            outputs.append(output)
        return outputs

class MiniGPT:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.temperature = 0
        self.top_p = 1
        self.model, self.template = load_indiv_model(model_name)
        self.n_input_tokens = 0
        self.n_output_tokens = 0
        self.n_input_chars = 0
        self.n_output_chars = 0

    def get_response(self, prompts_list: List[str], max_n_tokens=None, temperature=None, no_template=False) -> List[dict]:
        batchsize = len(prompts_list)
        tokenizer = self.model.tokenizer
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []  # batch of strings
        if no_template:
            full_prompts = prompts_list
        else:
            for conv, prompt in zip(convs_list, prompts_list):
                if 'mistral' in self.model_name:
                    # Mistral models don't use a system prompt so we emulate it within a user message
                    # following Vidgen et al. (2024) (https://arxiv.org/abs/2311.08370)
                    prompt = "SYSTEM PROMPT: Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.\n\n###\n\nUSER: " + prompt
                if 'llama3' in self.model_name or 'phi3' in self.model_name:
                    # instead of '[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n' for llama2
                    conv.system_template = '{system_message}'  
                if 'phi3' in self.model_name:
                    conv.system_message = 'You are a helpful AI assistant.'
                if "llama2" in self.model_name:
                    prompt = prompt + ' '
                conv.append_message(conv.roles[0], prompt)

                if "gpt" in self.model_name:
                    full_prompts.append(conv.to_openai_api_messages())
                # older models
                elif "vicuna" in self.model_name:
                    conv.append_message(conv.roles[1], None) 
                    formatted_prompt = conv.get_prompt()
                    full_prompts.append(formatted_prompt)
                elif "llama2" in self.model_name:
                    conv.append_message(conv.roles[1], None) 
                    formatted_prompt = '<s>' + conv.get_prompt()
                    full_prompts.append(formatted_prompt)
                # newer models
                elif "r2d2" in self.model_name or "gemma" in self.model_name or "mistral" in self.model_name or "llama3" in self.model_name or "phi3" in self.model_name: 
                    conv_list_dicts = conv.to_openai_api_messages()
                    if 'gemma' in self.model_name or 'mistral' in self.model_name:
                        conv_list_dicts = conv_list_dicts[1:]  # remove the system message inserted by FastChat
                    full_prompt = tokenizer.apply_chat_template(conv_list_dicts, tokenize=False, add_generation_prompt=True)
                    full_prompts.append(full_prompt)
                else:
                    raise ValueError(f"To use {self.model_name}, first double check what is the right conversation template. This is to prevent any potential mistakes in the way templates are applied.")
        outputs = self.model.generate(full_prompts, 
                                      max_n_tokens=max_n_tokens,  
                                      temperature=self.temperature if temperature is None else temperature,
                                      top_p=self.top_p
        )
        
        self.n_input_tokens += sum(output['n_input_tokens'] for output in outputs)
        self.n_output_tokens += sum(output['n_output_tokens'] for output in outputs)
        self.n_input_chars += sum(len(full_prompt) for full_prompt in full_prompts)
        self.n_output_chars += len([len(output['text']) for output in outputs])
        return outputs

    def generate(
        self, 
        convs: List[str], 
        max_n_tokens: int = 128, 
        temperature: float = 0.7, 
        top_p: float = 0.9
    ) -> List[Dict]:
        results = []
        for conv in convs:
            inputs = self.model.tokenizer(conv, return_tensors="pt").to(self.device)
            input_len = inputs.input_ids.shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_n_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            generated_ids = outputs.sequences[0][input_len:]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            scores = outputs.scores
            logprobs = []
            for i, score in enumerate(scores):
                probs = torch.log_softmax(score[0], dim=-1)
                token_id = generated_ids[i].item()
                topk_logprobs = torch.topk(probs, k=20)
                logprobs.append({
                    self.tokenizer.decode([tid]): lprob.item()
                    for tid, lprob in zip(topk_logprobs.indices, topk_logprobs.values)
                })

            results.append({
                'text': text,
                'logprobs': logprobs,
                'n_input_tokens': input_len,
                'n_output_tokens': len(generated_ids),
            })
        return results

class HuggingFace:
    def __init__(self,model_name, model, tokenizer):
        self.model_name = model_name
        self.model = model 
        self.tokenizer = tokenizer
        # substitute '▁Sure' with 'Sure' (note: can lead to collisions for some target_tokens)
        self.pos_to_token_dict = {v: k.replace('▁', ' ') for k, v in self.tokenizer.get_vocab().items()}
        # self.pos_to_token_dict = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.eos_token_ids = [self.tokenizer.eos_token_id]
        if 'llama3' in self.model_name.lower():
            self.eos_token_ids.append(self.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        self.pad_token_id = self.tokenizer.pad_token_id
    
    def generate(
    self,
    full_prompts_list: List[str],
    max_n_tokens: int,
    temperature: float,
    top_p: float = 1.0,
    ) -> List[Dict]:

        batch_size = len(full_prompts_list)

        # ---- tokenizer / pad 处理（一次性，鲁棒）----
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # ---- lazy cache：只构建一次 vocab 逆映射 ----
        if not hasattr(self, "_id_to_token"):
            vocab = self.tokenizer.get_vocab()
            self._id_to_token = [None] * len(vocab)
            for tok, idx in vocab.items():
                self._id_to_token[idx] = tok

        inputs = self.tokenizer(
            full_prompts_list,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        inputs = {k: v.to(self.model.device, non_blocking=True) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        # ---- generate ----
        output = self.model.generate(
            **inputs,
            max_new_tokens=max_n_tokens,
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else None,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            output_scores=True,
            return_dict_in_generate=True,
        )

        output_ids = output.sequences

        if not self.model.config.is_encoder_decoder:
            output_ids = output_ids[:, input_ids.shape[1]:]

        # ---- Llama2 / Vicuna 特判（鲁棒 + 低成本）----
        is_llama2 = (
            "llama-2" in self.model_name.lower()
            or "vicuna" in self.model_name.lower()
        )

        if is_llama2 and output_ids.size(1) > 0 and output_ids[0, 0].item() == 29871:
            output_ids = output_ids[:, 1:]
            scores = output.scores[1:]
        else:
            scores = output.scores

        generated_texts = self.tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        # ---- logits → logprobs（一次性转 CPU，避免隐式 sync）----
        # scores: [T][B][V]
        scores_cpu = [s.float().cpu() for s in scores]

        logprobs_tokens = [
            torch.nn.functional.log_softmax(s, dim=-1).numpy()
            for s in scores_cpu
        ]

        vocab_size = logprobs_tokens[0].shape[-1]
        id_to_token = self._id_to_token

        # ---- 构建 logprob dict（无分支、无重复构建）----
        logprob_dicts = [
            [
                {
                    id_to_token[i_vocab]:
                    logprobs_tokens[t][b][i_vocab]
                    for i_vocab in range(vocab_size)
                }
                for t in range(len(logprobs_tokens))
            ]
            for b in range(batch_size)
        ]

        n_input_tokens = (
            input_ids != self.tokenizer.pad_token_id
        ).sum(dim=1).tolist()

        outputs = [
            {
                "text": generated_texts[b],
                "logprobs": logprob_dicts[b],
                "n_input_tokens": n_input_tokens[b],
                "n_output_tokens": output_ids.shape[1],
            }
            for b in range(batch_size)
        ]

        # ---- 显存清理（最小必要集）----
        del output, scores_cpu, logprobs_tokens
        for k in inputs:
            inputs[k] = inputs[k].cpu()
        output_ids = output_ids.cpu()
        gc.collect()
        torch.cuda.empty_cache()

        return outputs


    # def generate(self, 
    #              full_prompts_list: List[str],
    #              max_n_tokens: int, 
    #              temperature: float,
    #              top_p: float = 1.0) -> List[Dict]:
    #     if 'llama2' in self.model_name.lower():
    #         max_n_tokens += 1  # +1 to account for the first special token (id=29871) for llama2 models
    #     batch_size = len(full_prompts_list)
    #     # vocab_size = len(self.tokenizer.get_vocab())
    #     vocab_size = self.tokenizer.vocab_size
    #     inputs = self.tokenizer(full_prompts_list, return_tensors='pt', padding=True)
    #     inputs = {k: v.to(self.model.device.index) for k, v in inputs.items()}
    #     input_ids = inputs["input_ids"]

    #     # Batch generation
    #     output = self.model.generate(
    #         **inputs,
    #         max_new_tokens=max_n_tokens,  
    #         do_sample=False if temperature == 0 else True,
    #         temperature=None if temperature == 0 else temperature,
    #         eos_token_id=self.eos_token_ids,
    #         pad_token_id=self.tokenizer.pad_token_id,  # added for Mistral
    #         top_p=top_p,
    #         output_scores=True,
    #         return_dict_in_generate=True,
    #     )
    #     output_ids = output.sequences
    #     # If the model is not an encoder-decoder type, slice off the input tokens
    #     if not self.model.config.is_encoder_decoder:
    #         output_ids = output_ids[:, input_ids.shape[1]:]  
    #     if 'llama2' in self.model_name.lower():
    #         output_ids = output_ids[:, 1:]  # ignore the first special token (id=29871)

    #     generated_texts = self.tokenizer.batch_decode(output_ids)
    #     # output.scores: n_output_tokens x batch_size x vocab_size (can be counter-intuitive that batch_size doesn't go first)
    #     logprobs_tokens = [torch.nn.functional.log_softmax(output.scores[i_out_token], dim=-1).cpu().numpy() 
    #                        for i_out_token in range(len(output.scores))]
    #     if 'llama2' in self.model_name.lower():
    #         logprobs_tokens = logprobs_tokens[1:]  # ignore the first special token (id=29871)

    #     logprob_dicts = [[{self.pos_to_token_dict[i_vocab]: logprobs_tokens[i_out_token][i_batch][i_vocab]
    #                      for i_vocab in range(vocab_size)} 
    #                      for i_out_token in range(len(logprobs_tokens))
    #                     ] for i_batch in range(batch_size)]

    #     outputs = [{'text': generated_texts[i_batch],
    #                 'logprobs': logprob_dicts[i_batch],
    #                 'n_input_tokens': len(input_ids[i_batch][input_ids[i_batch] != 0]),  # don't count zero-token padding
    #                 'n_output_tokens': len(output_ids[i_batch]),
    #                } for i_batch in range(batch_size)
    #     ]

    #     for key in inputs:
    #         inputs[key].to('cpu')
    #     output_ids.to('cpu')
    #     del inputs, output_ids
    #     gc.collect()
    #     torch.cuda.empty_cache()

    #     return outputs

    