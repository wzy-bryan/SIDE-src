DATA=(refine_prompts_50.csv)  # advbench subset
# DATA=(remove_guide_refine_prompts_50.csv)  # w/0 PGT
# DATA=(refine_reda_120.csv)  # reda dataset
# DATA=(prompt_50_remove_JSF.csv)    # w/0 JSF
# DATA=(refine_jailbreakbench_prompt_100.csv)  # jailbreakbench


echo "attack qwen2_5"
python3 main.py --target-model qwen2_5 --data-path dataset/side/${DATA} --logits-method EP+DGLM --lambda 50

echo "attack llama2-7b" 
python3 main.py --target-model llama2-7b --data-path dataset/side/${DATA} --logits-method EP+DGLM --lambda 50

echo "attack vicuna"
python3 main.py --target-model vicuna --data-path dataset/side/${DATA} --logits-method EP+DGLM --lambda 50

echo "attack gemma2-9b"
python3 main.py --target-model gemma2-9b --data-path dataset/side/${DATA} --logits-method EP+DGLM --lambda 50

echo "attack llama3-8b"
python3 main.py --target-model llama3-8b --data-path dataset/side/${DATA} --logits-method EP+DGLM --lambda 50

