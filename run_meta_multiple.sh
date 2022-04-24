task=$1
outdir=$2
device=$3
start=$4
end=$5
metric=$6 # Use {accuracy} for chemprot and hyperpartisan and {f1} for citation_intent and sciie

for k in $(seq $start $end)
do
	echo 'Working on Seed = '$k
	mkdir -p $outdir/$task/meta_tartan/seed=$k
	CUDA_VISIBLE_DEVICES=$device python -u -m scripts.run_mlm_auxiliary --train_data_file datasets/$task/train.txt --aux-task-names TAPT-MLM --line_by_line --model_type roberta-base --tokenizer_name roberta-base --mlm --model_name_or_path roberta-base --do_train --learning_rate 1e-4 --block_size 512 --logging_steps 5000 --classf_lr 1e-3 --primary_task_id $task -weight-strgy 'meta' --classifier_dropout 0.1 --classf_ft_lr 5e-6 --classf_max_seq_len 512 --classf-metric $metric --lazy-dataset --output_dir $outdir/$task/meta_tartan/seed=$k --overwrite_output_dir --seed $k  --classf_dev_wd 0.1 --classf_dev_lr 1e-3 --meta-lr-weight 1e-2 --dev_batch_sz 32 --classf_patience 20 --num_train_epochs 150 --classf_iter_batchsz 64 --per_gpu_train_batch_size 64 --gradient_accumulation_steps 2 --eval_every 30 --tapt-primsize --classf_warmup_frac 0.06 &> $outdir/$task/meta_tartan/seed=$k.txt
done

