predictory="zz1424:s3://visual_grounding/events/data/"
model='/mnt/lustre/share_data/zhangzhao2/VG/OFA_official_ckpt/hf_ofa_base'
tokenizer_path='/mnt/cache/zhangzhao2/codes/ofa-hf/tokenizer'
save_dir='/mnt/lustre/share_data/zhangzhao2/VG/log/hf-ofa-base/events_gen'
beam=5
min_len=1
no_repeat_size=3


echo "Model checkpoint is ${model}"
echo "Directory to save is ${save_dir}"
mkdir -p $save_dir


echo "smoke easy"
ref="person with a cigarrete close to the mouth"
val_dir='/mnt/lustre/share_data/zhangzhao2/VG/events/anno/smoke/val_easy.jsonl'
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

echo "business_tricycle"
ref="stall on a tricycle"
val_dir='/mnt/lustre/share_data/zhangzhao2/VG/events/anno/business_tricycle/val.jsonl'
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

echo "carry3people"
ref="motorbike with three people"
val_dir='/mnt/lustre/share_data/zhangzhao2/VG/events/anno/carry3people/val.jsonl'
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

echo "clothes_sheets"
ref="hanging clothes or sheets"
val_dir='/mnt/lustre/share_data/zhangzhao2/VG/events/anno/clothes_sheets/val.jsonl'
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

echo "trash"
ref="overfilled bin"
val_dir='/mnt/lustre/share_data/zhangzhao2/VG/events/anno/trash/val.jsonl'
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

echo "broken_tree"
ref="broken or fallen tree"
val_dir='/mnt/lustre/share_data/zhangzhao2/VG/events/anno/broken_tree/val.jsonl'
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

echo "fisherman"
ref="man fishing"
val_dir='/mnt/lustre/share_data/zhangzhao2/VG/events/anno/fisherman/val.jsonl'
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



echo "refclef"
ref="null ref"
val_dir='/mnt/lustre/share_data/zhangzhao2/VG/OFA_ceph_annos/ofa_eval_vg_refclef.jsonl'
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