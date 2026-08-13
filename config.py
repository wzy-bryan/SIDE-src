import os

# Project root (can be overridden with environment variable SIDE_ROOT)
PROJECT_ROOT = os.environ.get('SIDE_ROOT', os.path.abspath(os.path.dirname(__file__)))

# Paths to models: can be overridden with environment variables for portability
MODEL_ROOT = os.environ.get('LLM_MODEL_ROOT', '/mnt/data/home/wangzhiyuan/LLM-model')
VICUNA_PATH = os.environ.get('VICUNA_PATH', os.path.join(MODEL_ROOT, 'Vicuna-7b-v1.5/AI-ModelScope/vicuna-7b-v1.5'))
LLAMA_7B_CHAT_PATH = os.environ.get('LLAMA_7B_CHAT_PATH', os.path.join(MODEL_ROOT, 'Llama2-7b-chat', 'Llama-2-7b-chat-hf'))
LLAMA_7B_PATH = os.environ.get('LLAMA_7B_PATH', '/mnt/data/home/wangzhiyuan/LLM-model/Llama2-7b-chat-hf-v2/shakechen/Llama-2-7b-chat-hf')
LLAMA_13B_PATH = os.environ.get('LLAMA_13B_PATH', 'meta-llama/Llama-2-13b-chat-hf')
LLAMA_70B_PATH = os.environ.get('LLAMA_70B_PATH', 'meta-llama/Llama-2-70b-chat-hf')
LLAMA3_8B_PATH = os.environ.get('LLAMA3_8B_PATH', os.path.join(MODEL_ROOT, 'Llama3-8b-it/LLM-Research/Meta-Llama-3-8B-Instruct'))
LLAMA3_1_8B_PATH = os.environ.get('LLAMA3_1_8B_PATH', os.path.join(MODEL_ROOT, 'Llama3.1-8b-it'))
LLAMA3_70B_PATH = os.environ.get('LLAMA3_70B_PATH', 'meta-llama/Meta-Llama-3-70B-Instruct')
GEMMA_2B_PATH = os.environ.get('GEMMA_2B_PATH', os.path.join(MODEL_ROOT, 'gemma2-2b-it/LLM-Research/gemma-2-2b-it'))
GEMMA_7B_PATH = os.environ.get('GEMMA_7B_PATH', 'google/gemma-7b-it')
GEMMA_9B_PATH = os.environ.get('GEMMA_9B_PATH', os.path.join(MODEL_ROOT, 'gemma2-9b-it/LLM-Research/gemma-2-9b-it'))
MISTRAL_7B_PATH = os.environ.get('MISTRAL_7B_PATH', os.path.join(MODEL_ROOT, 'Mistral-7B-v0.1'))
MIXTRAL_7B_PATH = os.environ.get('MIXTRAL_7B_PATH', 'mistralai/Mixtral-8x7B-Instruct-v0.1')
R2D2_PATH = os.environ.get('R2D2_PATH', 'cais/zephyr_7b_r2d2')
PHI3_MINI_PATH = os.environ.get('PHI3_MINI_PATH', 'microsoft/Phi-3-mini-128k-instruct')
MINIGPT_PATH = os.environ.get('MINIGPT_PATH', os.path.join(MODEL_ROOT, 'MiniGpt-4'))
DEEPSEEK_PATH = os.environ.get('DEEPSEEK_PATH', os.path.join(MODEL_ROOT, 'deepseek', 'DeepSeek-R1-Distill-Qwen-1.5B'))

QWEN2_5_7B_PATH = os.environ.get('QWEN2_5_7B_PATH', os.path.join(MODEL_ROOT, 'Qwen2.5-7B-Instruct/Qwen/Qwen2.5-7B-Instruct'))

# Default generation settings
TARGET_TEMP = float(os.environ.get('TARGET_TEMP', 0))
TARGET_TOP_P = float(os.environ.get('TARGET_TOP_P', 1))
