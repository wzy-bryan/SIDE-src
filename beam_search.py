import numpy as np
import heapq
from judges import load_judge, judge_rule_based
from language_models import GPT
from prompts import get_universal_manual_prompt
from conversers import load_target_model, load_judge_model, judge_response
from utils import insert_adv_string, schedule_n_to_change_fixed, schedule_n_to_change_prob, extract_logprob, early_stopping_condition
import time
import heapq
import torch
import random
import torch.nn.functional as F
import torch
from conversers import *

def manipulate_none_token_then_generate(
        beam_adv: str,
        orig_msg: str,
        targetLM: TargetLM,
        tokenizer,
        target_token: str,          # 可以是任意长度，如 "Sure, here is"
        max_n_tokens: int,
        device: str = "cuda"
    ):
    """
    把 target_token 逐 token 地“硬控”进生成序列：
    每一步都把当前 target token 的 logit 与最大 logit 交换，然后贪心采样。
    控完 target 序列后，再自由生成剩余部分。
    """
    import torch

    # 1. 原始 prompt 编码
    prompt = orig_msg + beam_adv
    prompt_text = prompt
    print("prompt_text:", prompt_text)
  
    temperature = 0
    top_p=0.9
    # outputs = targetLM.get_response([prompt_text], max_n_tokens=max_n_tokens, temperature=0)
    orig_response_text = targetLM.get_response([prompt_text], max_n_tokens=max_n_tokens, temperature=0)[0]['text']
    # orig_response_text = targetLM.model.generate([prompt_text], 
    #                                   max_n_tokens=max_n_tokens,  
    #                                   temperature=temperature if temperature is None else temperature,
    #                                   top_p=top_p
    #     )[0]['text']
    
    # final_text = outputs[0]["text"]

    print("Manipulated generation:", orig_response_text)
    return orig_response_text

def model_response(
        full_prompt: str,
        targetLM: TargetLM,
        tokenizer,
        max_n_tokens: int,
        device: str = "cuda"
    ):
    import torch
    temperature = 0.3
    top_p = 0.8
    prompt = full_prompt
    prompt_size = len(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    bsz = input_ids.shape[0]
    ctx_len = input_ids.shape[1]
    text_trace = []  # 记录每一步生成的文本
    with torch.no_grad():
        for _ in range(max_n_tokens):
            if _ == 0:
                out = targetLM.model.model(
                        input_ids=input_ids,
                        use_cache=True
                    )
            else:
                out = targetLM.model.model(
                        input_ids=input_ids[:, -1:],          # 只喂最后一个 token
                        past_key_values=out.past_key_values,  # 携带上文
                        use_cache=True
                    )
            logits = out.logits[:, -1, :]          # [1, vocab]
            
            if temperature > 0:
                # softmax + top-p 采样
                probs = F.softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum_probs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                row_sum = sorted_probs.sum(dim=-1, keepdim=True)
                row_sum[row_sum == 0] = 1.0  # 避免除零
                sorted_probs.div_(row_sum)
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_id = torch.gather(sorted_idx, -1, next_token).reshape(-1)
            else:
                # 3.3 贪心采样（现在 tgt_id 已经是最大）
                next_id = torch.argmax(logits, dim=-1)  # [1, 1]
               
            # next_id = logits.argmax(dim=-1, keepdim=True)  # 贪心采样
            input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=1)

            # 记录当前文本
            text_trace.append(tokenizer.decode(input_ids[0][ctx_len:]))
    
    final_text = text_trace[-1] if text_trace else ""
    return final_text


def ppl(
        full_prompt: str,
        targetLM: TargetLM,
        tokenizer,
        max_n_tokens: int,
        device: str = "cuda"
    ):
    """
    计算给定 prompt 的困惑度（perplexity）。接口与 :func:`model_response`
    保持一致，这样调用时可以直接互换。

    这里只对输入的 ``full_prompt`` 进行打分，不使用 ``max_n_tokens``
    参数（仅为了保持签名一致）。

    返回值为一个浮点数，表示模型在该 prompt 上的 PPL。
    如果需要针对生成后续文本的 PPL，可在外部先拼接文本再调用本函数。
    """

    import torch

    # 编码 prompt
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

    # 直接利用 HuggingFace API 计算交叉熵损失，loss 是平均的负对数似然
    # CausalLM 的实现会自动把 labels 向右移动一位，因此我们可以把 labels
    # 设为 input_ids 本身。
    with torch.no_grad():
        outputs = targetLM.model.model(input_ids, labels=input_ids)
        # 有些模型返回 tuple，第一个元素是 loss
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]

    # perplexity = exp(loss)
    try:
        ppl_val = float(torch.exp(loss))
    except Exception:
        # 如果 loss 已经是 python float
        ppl_val = float(np.exp(loss))
    return ppl_val
    

def manipulate_firstfew_token_then_generate(
        full_prompt: str,
        targetLM: TargetLM,
        tokenizer,
        target_token: str,          # 可以是任意长度，如 "Sure, here is"
        max_n_tokens: int,
        device: str = "cuda"
    ):
    
    """
    把 target_token 逐 token 地“硬控”进生成序列：
    每一步都把当前 target token 的 logit 与最大 logit 交换，然后贪心采样。
    控完 target 序列后，再自由生成剩余部分。
    """
    import torch
    temperature=0.0
    top_p= 0.8
    # 1. 原始 prompt 编码
    prompt = full_prompt
    prompt_size = len(prompt)
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    bsz = input_ids.shape[0]
    ctx_len = input_ids.shape[1]
    assert bsz == 1, "下面逻辑只支持 batch=1"

    # 2. 把 target_token 编码成 id 序列
    sure_ids = tokenizer.encode("Sure", add_special_tokens=False)
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    k = len(target_ids)
    target_ids = torch.tensor(target_ids, device=device).unsqueeze(0)  # [1, k]

    # 3. 逐位置“硬控”
    past_key_values = None
    for i in range(k):
        # 3.1 取当前最后一个 token 的 logits
        with torch.no_grad():
            if i == 0:
                out = targetLM.model.model(
                        input_ids=input_ids,
                        use_cache=True
                    )
            else:
                out = targetLM.model.model(
                        input_ids=input_ids[:, -1:],          # 只喂最后一个 token
                        past_key_values=out.past_key_values,  # 携带上文
                        use_cache=True
                    )
            # out = targetLM.model.model(
            #     input_ids=input_ids[:, -1:],      # 只喂最后一个 token，加速
            #     past_key_values=past_key_values,
            #     use_cache=True
            # )
            logits = out.logits[:, -1, :]        # [1, vocab]
            past_key_values = out.past_key_values

        # 3.2 把 target_ids[0, i] 的 logit 与最大 logit 交换
        tgt_id = target_ids[0, i]
        max_id = logits.argmax(dim=-1)[0]
        tmp = logits[..., tgt_id].clone()
        
        if tgt_id.item() != max_id.item():
            tmp = logits[..., tgt_id].clone()
            logits[..., tgt_id] = logits[..., max_id]
            logits[..., max_id] = tmp
            # logits[..., tgt_id], logits[..., max_id] = logits[..., max_id].clone(), logits[..., tgt_id].clone()

        if temperature > 0:
            # softmax + top-p 采样
            probs = F.softmax(logits / temperature, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
            mask = cumsum_probs - sorted_probs > top_p
            sorted_probs[mask] = 0.0
            row_sum = sorted_probs.sum(dim=-1, keepdim=True)
            row_sum[row_sum == 0] = 1.0  # 避免除零
            sorted_probs.div_(row_sum)
            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_id = torch.gather(sorted_idx, -1, next_token).reshape(-1)
        else:
            # 3.3 贪心采样（现在 tgt_id 已经是最大）
            next_id = torch.argmax(logits, dim=-1)  # [1, 1]
            if logits[..., next_id] == logits[..., tgt_id]:
                next_id = tgt_id.unsqueeze(0)
            else:
                assert next_id.item() == tgt_id.item(), "swap 后贪心必须选中 target token"

        # 3.4 更新输入
        input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=1)

    # 4. 控完 target 序列，用 get_response 继续自由生成
    # prompt_text = tokenizer.decode(input_ids[0][ctx_len:], skip_special_tokens=True)
    # print("prompt_text after manipulation:", prompt_text)

    remain = max_n_tokens - k

    # 4. 继续用 model.model 自回归生成剩余部分
    already_generated = k
    remain = max_n_tokens - already_generated
    text_trace = []  # 记录每一步生成的文本，方便调试
    top3_tokens = []   # 每个元素是 list[int]，长度 5
    top3_probs  = []   # 每个元素是 list[float]，长度 5
    temperature=0.3
    with torch.no_grad():
        for _ in range(remain):
            if _ == 0:
                out = targetLM.model.model(
                        input_ids=input_ids,
                        use_cache=True
                    )
            else:
                out = targetLM.model.model(
                        input_ids=input_ids[:, -1:],          # 只喂最后一个 token
                        past_key_values=out.past_key_values,  # 携带上文
                        use_cache=True
                    )
            # out = targetLM.model.model(
            #     input_ids=input_ids[:, -1:],
            #     past_key_values=past_key_values,
            #     use_cache=True
            # )
            logits = out.logits[:, -1, :]          # [1, vocab]
            probs_temp = torch.softmax(logits, dim=-1)
            top3_p, top3_idx = torch.topk(probs_temp, k=3, dim=-1)   # 都是 [1, 5]
            top3_tokens.append(top3_idx[0].cpu().tolist())      # list[int]
            top3_probs.append(top3_p[0].cpu().tolist())         # list[float]

            # if _ == 1:
            #     #sort logits,get top 4 logits token:
            #     top4_logits, top4_idx = torch.topk(probs_temp, k=5, dim=-1)   # 都是 [1, 5]
            #     tgt_id = sure_ids
            #     print("Sure probs: ", logits[0, tgt_id].item())
            #     print("Top 5 probs tokens for first generated token:", [(tokenizer.decode([top4_idx[0][i].item()]), top4_logits[0][i].item()) for i in range(5)])
            # past_key_values = out.past_key_values
            
            if temperature > 0:
                # softmax + top-p 采样
                probs = F.softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum_probs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                row_sum = sorted_probs.sum(dim=-1, keepdim=True)
                row_sum[row_sum == 0] = 1.0  # 避免除零
                sorted_probs.div_(row_sum)
                next_token = torch.multinomial(sorted_probs, num_samples=1)
                next_id = torch.gather(sorted_idx, -1, next_token).reshape(-1)
            else:
                # 3.3 贪心采样（现在 tgt_id 已经是最大）
                next_id = torch.argmax(logits, dim=-1)  # [1, 1]
                # assert next_id.item() == tgt_id.item(), "swap 后贪心必须选中 target token"
            # next_id = logits.argmax(dim=-1, keepdim=True)  # 贪心采样
            input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=1)

            # 记录当前文本
            text_trace.append(tokenizer.decode(input_ids[0][ctx_len:]))
    
    final_text = text_trace[-1] if text_trace else ""
    # final_text = outputs[0]["text"]
    top3_tokens_first10 = top3_tokens[:10]
    top3_probs_first10  = top3_probs[:10]
    # top3_text_first10 = [[tokenizer.decode(tid, skip_special_tokens=True) for tid in step]for step in top3_tokens_first10]
    # for step in top3_tokens_first10:
    #     print("Step:", step)
    #     for tid in step:
    #         print("tid: ", tid)
    #         print(f"  {tokenizer.decode([tid], skip_special_tokens=True):>12} ", end="")
    top3_text_first10 = [[tokenizer.decode([tid]) for tid in step]for step in top3_tokens_first10]
    
    # for step, (tokens, probs) in enumerate(zip(top3_tokens_first10, top3_probs_first10)):
    #     print(f"Step {step}:")
    #     for tok, p in zip(tokens, probs):
    #         print(f"  {tokenizer.decode(tok):>12}  {p:.4f}")

    # print("Manipulated generation:", final_text)
    return final_text, top3_text_first10, top3_probs_first10

def manipulate_first_token_then_generate(
        beam_adv: str,
        orig_msg: str,
        targetLM,            # TargetLM 实例
        tokenizer,
        target_token: str,   # 字符串形式
        max_n_tokens: int,
        device: str = "cuda"
    ):
    """
    仅对第一个新生成 token 做 logits 替换，之后直接调用 model.generate 完成后续文本。
    返回模型生成的完整回复（不含 prompt）。
    """

    # 1. 构造 prompt → tokenize
    prompt = orig_msg + beam_adv
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)
    bsz, prompt_len = input_ids.shape

    # 2. 获取 target_token 对应的 id
    target_token_id = tokenizer.convert_tokens_to_ids(target_token)
    if target_token_id is None:
        raise ValueError(f"target_token '{target_token}' 不在词表中！")

    # 3. 只跑前向拿到最后一个 logits，手动替换
    with torch.inference_mode():
        outputs = targetLM.model(
            input_ids=input_ids,
            use_cache=True
        )
        logits = outputs.logits[:, -1, :]          # [bsz, vocab]
        past_key_values = outputs.past_key_values  # 后续供 generate 复用 KV-cache

        # 交换：max-logit token <-> target_token
        max_idx = logits.argmax(dim=-1)
        batch_range = torch.arange(bsz, device=device)
        logits[batch_range, max_idx], logits[batch_range, target_token_id] = \
            logits[batch_range, target_token_id], logits[batch_range, max_idx].clone()

        # 贪婪选第一个 token
        first_token = logits.argmax(dim=-1, keepdim=True)  # [bsz, 1]

    # 4. 用 generate 一次性完成剩余 max_n_tokens-1 个 token
    gen_out = targetLM.model.generate(
        input_ids=first_token,           # 从第一个被操纵的 token 开始
        past_key_values=past_key_values, # 复用 KV-cache
        attention_mask=torch.ones_like(first_token),
        max_new_tokens=max_n_tokens - 1,
        do_sample=False,                 # 贪婪
        pad_token_id=tokenizer.eos_token_id
    )

    # 5. 拼接并解码
    #    gen_out 形状: [bsz, 1 + (max_n_tokens-1)] = [bsz, max_n_tokens]
    full_ids = torch.cat([input_ids, gen_out], dim=1)
    new_ids  = full_ids[:, prompt_len:]          # 只取模型生成的部分
    text = tokenizer.decode(new_ids[0])
    print(f"Manipulated generation (first-token-only): {text}")
    return text


def manipulate_output_token_probability(
    beam_adv,
    orig_msg,
    targetLM,
    tokenizer,
    target_token_string,
    max_n_tokens
):
    """
    拼接 prompt -> forward 得到 logits -> 
    第一个 token 修改 target_token 的概率分布 -> softmax+采样 -> 
    后续 token 正常 softmax 采样
    """
    temperature = 1.0
    top_p = 0.8
    device = targetLM.device if hasattr(targetLM, "device") else "cuda" if torch.cuda.is_available() else "cpu"
    # 拼接完整 prompt
    prompt = orig_msg + beam_adv
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    bsz = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    # 获取 target token ID
    target_token_id = tokenizer.encode(target_token_string, add_special_tokens=False)[0]

    # 初始化生成序列
    generated_ids = input_ids.clone()
    past_key_values = None

    with torch.inference_mode():
        # 第一个 token
        outputs = targetLM.model.model(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # [bsz, vocab_size]

        # 交换 target_token 和最大 logit token
        max_idx = torch.argmax(logits, dim=-1)  # 当前最大 logit token
        batch_indices = torch.arange(bsz, device=device)
        temp = logits[batch_indices, max_idx].clone()
        logits[batch_indices, max_idx] = logits[batch_indices, target_token_id]
        logits[batch_indices, target_token_id] = temp

        # softmax + top-p 采样
        probs = F.softmax(logits / temperature, dim=-1)
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumsum_probs - sorted_probs > top_p
        sorted_probs[mask] = 0.0
        row_sum = sorted_probs.sum(dim=-1, keepdim=True)
        row_sum[row_sum == 0] = 1.0  # 避免除零
        sorted_probs.div_(row_sum)
        next_token = torch.multinomial(sorted_probs, num_samples=1)
        next_token = torch.gather(sorted_idx, -1, next_token).reshape(-1)
        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)

        # 后续 token
        for _ in range(max_n_tokens - 1):
            input_step = generated_ids[:, -1:].to(device)
            outputs = targetLM.model.model(
                input_ids=input_step,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).reshape(-1)
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)

            # 释放显存
            del logits, probs, input_step
            torch.cuda.empty_cache()

    # 只取模型生成的部分
    gen_ids_only = generated_ids[:, prompt_len:]
    final_text = tokenizer.decode(gen_ids_only[0])
    print(f"Manipulated generation: {final_text}")
    return final_text



import torch

def manipulate_first_token_then_generate(
        beam_adv: str,
        orig_msg: str,
        targetLM,       # TargetLM 实例
        tokenizer,
        target_token: str,  # 字符串形式
        max_n_tokens: int,
        device: str = "cuda"
    ):
    """
    对第一个 token 替换 logits，后续 token 用模型 generate 或 get_response 生成。
    返回模型生成的响应部分。
    """

    # 拼接完整 prompt
    prompt = orig_msg + beam_adv
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    bsz = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    # 获取 target token ID
    target_token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]

    # 第一步：获取第一个 token logits
    with torch.inference_mode():
        outputs = targetLM.model.model(
            input_ids=input_ids,
            use_cache=True
        )
        logits = outputs.logits[:, -1, :]  # [bsz, vocab_size]

        # 交换 target_token 和最大 logit token
        max_idx = torch.argmax(logits, dim=-1)
        batch_indices = torch.arange(bsz, device=device)
        temp = logits[batch_indices, max_idx].clone()
        logits[batch_indices, max_idx] = logits[batch_indices, target_token_id]
        logits[batch_indices, target_token_id] = temp

        # 贪婪选择第一个 token
        next_token = torch.argmax(logits, dim=-1).unsqueeze(1)

    # 拼接到输入序列，作为新的 prompt
    new_input_ids = torch.cat([input_ids, next_token], dim=1)

    # 后续 token 生成
    prompt_text = tokenizer.decode(new_input_ids[0], skip_special_tokens=True)
    print("prompt_text: ", prompt_text)
    outputs = targetLM.get_response(
        [prompt_text],
        max_n_tokens=max_n_tokens - 1
    )

    final_text = outputs[0]["text"]

    # 去掉原始 prompt，只保留模型生成的响应
    # 找到 prompt+第一个token在 final_text 中的位置
    prompt_plus_first_token = tokenizer.decode(new_input_ids[0], skip_special_tokens=True)
    if final_text.startswith(prompt_plus_first_token):
        final_text = final_text[len(prompt_plus_first_token):]

    

    print("Manipulated generation (first token modified):", final_text)
    return final_text



def beam_search_attack_random(
    orig_msg, target_token, targetLM, beam_size=5, n_iterations=50,
    lookahead_length=15, n_trials=1, sample_size=5, max_char_len=200
):
    print("Performing beam search with random sampling...")
    # 缓存：避免重复调用 LM
    cache = {}

    best_adv = ""
    best_msg = insert_adv_string(orig_msg, best_adv)
    best_logprob = -np.inf
    best_logprobs, best_advs = [], []

    # 第一次请求（初始化 beam）
    msg = insert_adv_string(orig_msg, "")
    if msg not in cache:
        output = targetLM.get_response([msg], max_n_tokens=1)[0]
        cache[msg] = output['logprobs'][0]
    logprob_dict = cache[msg]

    # 从所有 token 里随机采样 sample_size 个
    all_tokens = list(logprob_dict.items())
    sampled_items = random.sample(all_tokens, min(sample_size, len(all_tokens)))
    # 先随便取前 beam_size 个作为初始 beam（没有排序逻辑）
    beam = [(token_str, logp) for token_str, logp in sampled_items[:beam_size]]
    print("initial beam:", beam)

    for it in range(1, n_iterations + 1):
        candidates = []
        msgs = []
        adv_list = []

        for adv, _ in beam:
            msg = insert_adv_string(orig_msg, adv)
            if msg not in cache:
                output = targetLM.get_response([msg], max_n_tokens=1)[0]
                cache[msg] = output['logprobs'][0]
            logprob_dict = cache[msg]

            # 每个 beam 扩展时随机采样 sample_size 个 token
            all_tokens = list(logprob_dict.items())
            sampled_items = random.sample(all_tokens, min(sample_size, len(all_tokens)))

            for token_str, logp in sampled_items:
                new_adv = adv + token_str
                new_msg = insert_adv_string(orig_msg, new_adv)
                msgs.append(new_msg)
                adv_list.append(new_adv)

        # 批量 lookahead 请求
        outputs = targetLM.get_response(msgs, max_n_tokens=lookahead_length)

        for new_adv, out in zip(adv_list, outputs):
            score = 0
            for _ in range(n_trials):
                logprob2 = extract_logprob(out['logprobs'][0], target_token)
                score += logprob2
            score /= n_trials
            candidates.append((new_adv, score))

        # 更新 beam：保留得分最高的 beam_size 个
        candidates = heapq.nlargest(beam_size, candidates, key=lambda x: x[1])
        beam = candidates

        # 更新全局最佳
        if candidates[0][1] > best_logprob:
            best_adv, best_logprob = candidates[0]
            best_msg = insert_adv_string(orig_msg, best_adv)

        best_logprobs.append(best_logprob)
        best_advs.append(best_adv)

        print(f'it={it} [best] logprob={best_logprob:.3f} prob={np.exp(best_logprob):.5f} adv={best_adv} best_adv_len={len(best_adv)}')
        if len(best_adv) >= max_char_len:
            outputs = targetLM.get_response(msgs, max_n_tokens=lookahead_length)

            final_response_text = outputs[0]['text']
            break

    final_adv = best_adv
    final_msg = insert_adv_string(orig_msg, final_adv)
    outputs = targetLM.get_response([final_msg], max_n_tokens=lookahead_length)
    final_response_text = outputs[0]['text']

    return best_adv

def beam_search_attack(
    orig_msg, target_token, targetLM, beam_size=5, n_iterations=20,
    lookahead_length=15, n_trials=1
):
    # 缓存：避免重复调用 LM
    cache = {}

    best_adv = ""
    best_msg = insert_adv_string(orig_msg, best_adv)
    best_logprob = -np.inf
    best_logprobs, best_advs = [], []

    # 第一次请求（初始化 beam）
    msg = insert_adv_string(orig_msg, "")
    if msg not in cache:
        output = targetLM.get_response([msg], max_n_tokens=1)[0]
        cache[msg] = output['logprobs'][0]
    logprob_dict = cache[msg]

    # top-k 初始化
    sorted_items = heapq.nlargest(beam_size, logprob_dict.items(), key=lambda x: x[1])
    beam = [(token_str, logp) for token_str, logp in sorted_items]
    print("initial beam:", beam)

    for it in range(1, n_iterations + 1):
        candidates = []

        # -------- 批量化：收集所有待评估的扩展 --------
        msgs = []
        adv_list = []
        for adv, _ in beam:
            msg = insert_adv_string(orig_msg, adv)
            if msg not in cache:
                output = targetLM.get_response([msg], max_n_tokens=1)[0]
                cache[msg] = output['logprobs'][0]
            logprob_dict = cache[msg]

            # 扩展 top-k
            sorted_items = heapq.nlargest(beam_size, logprob_dict.items(), key=lambda x: x[1])
            for token_str, logp in sorted_items:
                new_adv = adv + token_str
                new_msg = insert_adv_string(orig_msg, new_adv)
                msgs.append(new_msg)
                adv_list.append(new_adv)

        # -------- 批量请求 lookahead --------
        outputs = targetLM.get_response(msgs, max_n_tokens=lookahead_length)

        for new_adv, out in zip(adv_list, outputs):
            score = 0
            for _ in range(n_trials):  # 多次 trial（可以设 n_trials=1 加速）
                logprob2 = extract_logprob(out['logprobs'][0], target_token)
                score += logprob2
            score /= n_trials
            candidates.append((new_adv, score))

        # -------- 更新 beam --------
        candidates = heapq.nlargest(beam_size, candidates, key=lambda x: x[1])
        beam = candidates

        # 更新全局最佳
        if candidates[0][1] > best_logprob:
            best_adv, best_logprob = candidates[0]
            best_msg = insert_adv_string(orig_msg, best_adv)

        best_logprobs.append(best_logprob)
        best_advs.append(best_adv)

        print(f'it={it} [best] logprob={best_logprob:.3f} prob={np.exp(best_logprob):.5f} adv={best_adv}')

    return best_adv


def _softmax(logits, temp=1.0):
    # logits: numpy array
    if temp != 1.0:
        logits = logits / float(temp)
    # 为数值稳定性减去最大值
    logits = logits - np.max(logits)
    exps = np.exp(logits)
    probs = exps / (exps.sum() + 1e-20)
    return probs

def sample_k_from_logprob_dict(logprob_dict, k, temperature=1.0):
    """
    logprob_dict: dict(token_str -> logprob)  # log probabilities (not logits) or scores
    k: number of samples to draw (without replacement)
    temperature: temperature applied to logits; higher -> more uniform
    Returns: list of (token_str, logprob) sampled (len <= k if vocab small)
    """
    # convert to lists
    tokens = list(logprob_dict.keys())
    logps = np.array([logprob_dict[t] for t in tokens], dtype=float)  # assume log-probabilities
    # Convert log-probs to logits for temperature scaling: logits = logp (since logp = log softmax)
    # We can treat logps as unnormalized logit-like scores (works fine for sampling after softmax)
    probs = _softmax(logps, temp=temperature)  # apply softmax(temp)
    # handle case k > vocab
    k_eff = min(k, len(tokens))
    # use torch.multinomial for without-replacement sampling
    probs_t = torch.tensor(probs, dtype=torch.double)
    with torch.no_grad():
        # replacement=False -> without replacement
        idx = torch.multinomial(probs_t, num_samples=k_eff, replacement=False).cpu().numpy().tolist()
    sampled = [(tokens[i], float(logps[i])) for i in idx]
    return sampled

def beam_search_attack_stochastic(
    orig_msg, target_token, targetLM, beam_size=5, n_iterations=20,
    lookahead_length=15, n_trials=1, sample_k=None, temperature=1.0
):
    """
    Stochastic (no-replacement multinomial) beam-search attack.
    sample_k: number of samples to draw per beam for expansion (if None, use beam_size)
    temperature: temperature applied when sampling from next-token distribution
    """
    if sample_k is None:
        sample_k = beam_size

    # cache to avoid repeated LM calls
    cache = {}

    best_adv = ""
    best_msg = insert_adv_string(orig_msg, best_adv)
    best_logprob = -np.inf
    best_logprobs, best_advs = [], []

    # 初次请求（初始化 beam）
    msg0 = insert_adv_string(orig_msg, "")
    if msg0 not in cache:
        output = targetLM.get_response([msg0], max_n_tokens=1)[0]
        # 假设 output['logprobs'][0] 是 dict(token_str -> logprob)
        cache[msg0] = output['logprobs'][0]
    logprob_dict = cache[msg0]

    # 用无放回采样初始化 beam（而不是 deterministic top-k）
    init_samples = sample_k_from_logprob_dict(logprob_dict, k=beam_size, temperature=temperature)
    # beam: list of tuples (adv_string, score_logprob_of_last_token)
    beam = [(token_str, logp) for (token_str, logp) in init_samples]
    print("initial beam (sampled):", beam)

    for it in range(1, n_iterations + 1):
        candidates = []
        # 批量化收集所有要评估的扩展（msgs 和对应的 new adv）
        msgs = []
        adv_list = []

        for adv, _last_lp in beam:
            msg = insert_adv_string(orig_msg, adv)
            if msg not in cache:
                output = targetLM.get_response([msg], max_n_tokens=1)[0]
                cache[msg] = output['logprobs'][0]
            logprob_dict = cache[msg]

            # 对每个 beam 用不放回的多项式采样得到 k 个 token 候选（而不是 top-k）
            sampled_items = sample_k_from_logprob_dict(logprob_dict, k=sample_k, temperature=temperature)
            for token_str, _lp in sampled_items:
                new_adv = adv + token_str
                new_msg = insert_adv_string(orig_msg, new_adv)
                msgs.append(new_msg)
                adv_list.append(new_adv)

        if len(msgs) == 0:
            break

        # 批量请求 lookahead（一次性得到每个 new_msg 的后续生成）
        outputs = targetLM.get_response(msgs, max_n_tokens=lookahead_length)

        # 对每个候选计算攻击分数（targeted 或 untargeted），这里假设 target_token 非 None -> targeted
        for new_adv, out in zip(adv_list, outputs):
            score = 0.0
            for _ in range(n_trials):
                # extract_logprob 函数负责从 out['logprobs'][0] 中读取 target_token 的 logprob
                # 如果是 untargeted attack，你可以替换成 perplexity-based score
                logprob2 = extract_logprob(out['logprobs'][0], target_token)
                score += logprob2
            score /= max(1, n_trials)
            candidates.append((new_adv, score))

        # 更新 beam：在每个原始 prompt（若有 batch）中按分段挑选 top beam_size
        # 原实现是对每个原始 prompt 分开处理（按 bs 切分），为兼容尽量保留同样的分割逻辑
        # 这里假定 targetLM.get_response 批量顺序与 adv_list 顺序一致，且 adv_list 是按照 beam 列表展平的
        # 如果 orig_msg 是单条（非 batch），则直接选择全局 top-k
        if isinstance(orig_msg, list):
            # 若支持 batch（多个 orig_msg），需要把 candidates 分块按每个 batch 分别选 top beam_size
            # 为简洁示范，这里假设单条 orig_msg；若你需要 batch，请告知我以便我增强分块逻辑
            pass

        # 选择分数最高的 beam_size 个候选作为新的 beam
        top_candidates = heapq.nlargest(beam_size, candidates, key=lambda x: x[1])
        beam = top_candidates

        # 更新全局最优
        if len(top_candidates) > 0 and top_candidates[0][1] > best_logprob:
            best_adv, best_logprob = top_candidates[0]
            best_msg = insert_adv_string(orig_msg, best_adv)

        best_logprobs.append(best_logprob)
        best_advs.append(best_adv)

        print(f'it={it} [best] logprob={best_logprob:.3f} prob={np.exp(best_logprob):.5f} adv={best_adv}')

    return best_adv, best_logprob, best_msg, best_logprobs, best_advs

