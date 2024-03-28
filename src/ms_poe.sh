# export HF_HOME="/playpen/xinyu"

# for file in "mdqa_results"/*; do
#     echo $file
#     python -u utils/lost_in_the_middle/eval_qa_response.py --input-path $file
# done

MODELNAME="meta-llama/Meta-Llama-3-8B-Instruct"
OUTPUTNAME="mdqa_results/llama3_8b-10doc-answer1-ratio1to1.jsonl.jsonl"
# OUTPUTNAME="mdqa_results/ours-mistral_7b-10doc-answer1-ratio1.2to1.8.jsonl"

CUDA_VISIBLE_DEVICES=0 python -u inference.py \
    --input_path data/mdqa_10documents.jsonl.gz \
    --output_path $OUTPUTNAME \
    --model_name $MODELNAME \
    --seed 42\
    --sample_num 5 \
    --answer_idx 1 \
    --enable_ms_poe \
    --apply_layers "2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30" \
    --compress_ratio_min 1 \
    --compress_ratio_max 1
python -u utils/lost_in_the_middle/eval_qa_response.py --input-path $OUTPUTNAME