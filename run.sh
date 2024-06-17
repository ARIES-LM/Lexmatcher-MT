

export CUDA_VISIBLE_DEVICES=0
ngpu=1

nshot=3

src=en
tgt=zh

src_file=prefix.$src
tgt_file=prefix.$tgt

python 01_filter_interactive.py $src_file $tgt_file

src_file_filtered=$src_file.clean_filt_ms
tgt_file_filtered=$tgt_file.clean_filt_ms

python 02_cometkiwi_sentpair.py  $src_file_filtered $tgt_file_filtered

python 03_sort_by_cometkiwi.py $src_file_filtered $tgt_file_filtered train.cometkiwi 0.5

bash token.sh

python 04_prelemma.py --token 0 --ds_path $src_file_filtered.kiwi.tok --tgt_path $tgt_file_filtered.kiwi.tok --source_lang $src --target_lang $tgt

align_path="wn_wik_bidict_$src$tgt.json"
align_type="wnwk"

python 05_select_training_data.py --nshot $nshot --token 0 --align_type $align_type --min_align_in_sent 1 --src_path $src_file_filtered.kiwi.tok --tgt_path $tgt_file_filtered.kiwi.tok --source_lang $src --target_lang $tgt --align_path $align_path --write_freq_path wik_freq$src$tgt.shot${nshot}.json >log.$src$tgt.$align_type.shot${nshot} 2>&1

bash token.sh detok $src $src_file_filtered.kiwi.$align_type.nshot${nshot}minali${minalign}
bash token.sh detok $tgt $tgt_file_filtered.kiwi.$align_type.nshot${nshot}minali${minalign}



