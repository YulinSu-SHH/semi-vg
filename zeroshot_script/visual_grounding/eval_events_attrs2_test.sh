predictory="zz1424:s3://visual_grounding/events/data/"
model='/mnt/lustre/share_data/zhangzhao2/VG/log/hf-ofa-base-attr-s2/pretrain/1016_1537/gs213022'
tokenizer_path='/mnt/cache/zhangzhao2/codes/ofa-hf/tokenizer'
save_dir='/mnt/lustre/share_data/zhangzhao2/VG/log/hf-ofa-base-attr-s2/events/test_gen'
beam=5
min_len=1
no_repeat_size=3

echo "Model checkpoint is ${model}"
echo "Directory to save is ${save_dir}"
mkdir -p $save_dir


echo "dog"
ref="dog without leash"
val_dir='/mnt/lustre/share_data/zhangzhao2/VG/events/anno/dog/val.jsonl'
echo "Run ofa visual grounding ${ref} model ${model} "
srun -p src_v100_32g -n1 -N 1 --gres=gpu:1 --job-name=ofa_hf_eval_pretrain_on_event --comment wbsR-SC221419.001 \
python ofa-base_vg_inference_generator.py \
--predictory ${predictory} \
--model ${model} \
--tokenizer_path=${tokenizer_path} \
--ref="${ref}" \
--beam=${beam} \
--min_len ${min_len} \
--no_repeat_size ${no_repeat_size} \
--val_dir ${val_dir} \
--save_dir ${save_dir} \
--gpu 0 &
