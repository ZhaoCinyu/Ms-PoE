
MODELNAME="meta-llama/Meta-Llama-3-8B-Instruct"
# MODELNAME="lmsys/vicuna-7b-v1.5"
# MODELNAME="Qwen/Qwen2-7B-Instruct"
OUTPUTNAME="mdqa_results/llama3_8b-10doc-answer1.jsonl"
# OUTPUTNAME="mdqa_results/ours-vicuna_7b-10doc-answer1.jsonl"
# OUTPUTNAME="mdqa_results/ours-qwen_7b-10doc-answer1-ratio1.2to1.8.jsonl"

CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path data/mdqa_10documents.jsonl.gz \
    --output_path $OUTPUTNAME \
    --model_name $MODELNAME \
    --seed 42 \
    --sample_num 5 \
    --answer_idx 1
python -u utils/lost_in_the_middle/eval_qa_response.py --input-path $OUTPUTNAME