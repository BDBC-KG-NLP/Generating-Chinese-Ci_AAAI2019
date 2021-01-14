This document intends to describe how to use the script to train, test and do experiments. I have divided into three sections and they are the specification of each script, explanation of the configuration and the instrusction of usage.
#####���ĵ�ּ���������ʹ�ýű�����ѵ�������Ժ���ʵ�顣�ҷ�Ϊ�������֣����Ƿֱ���ÿ���ű���˵�������ý�����ʹ��ָ��


## Specification of Scripts:
#####�ű��淶˵��
### Main Components:

config.py					model's config file
#####ģ�������ļ�
main.py						the main script to run our model
#####����ģ�����ű�
model.py					our model's basic modules eg.: GRU unit, encoder/decoder unit and VAE module
#####ģ�ͻ���ģ�� ���磺GRU��Ԫ������/����ģ�飬VAEģ��
trainer.py 					the trainer who use and connect all the modules from model.py in order to finish our XiaoQi task
#####ʹ�ò�����Model.py������ģ���ѵ�����������XiaoQi����


### Utility Components:
#####�������
iterator.py					take care of a batch of input(include feeding ci'vocab input,vowel and tone input)
#####����һ�����루�����ṩ�ʻ㣬Ԫ���Լ����������룩
rule.py						checking length, tone and rhyming foot
#####У�鳤�ȣ��������Ͻ�
ioHelper.py					read all preprocessing files
#####��ȡ����Ԥ�����ļ�
toneVowelHelper.py 			load tone and vowel input
#####����Ԫ����������
userHelper.py				testing for user's intents(*this is abandoned for a while ago... ) 
#####�û���ͼ���ԣ����Ѿ���������һ��ʱ���ˣ�
statistics_hardcoded.py		emperical distribution of Cipai and rhyming foot
#####���ɺ��Ͻŵľ���ֲ�
evaluation.py 				checking all three rules and see final generations when evaluating and testing
#####��������������򣬲��������Ͳ���ʱ�鿴��������
utils.py					functions for models'paramaters' path and vocab,tone,vowel lookup table.
#####ģ�Ͳ���·���ʹʻ㡢������Ԫ�����ұ�


### Extra:

version.sh  				checking pytorch and python's version
#####��ѯpytorch��python�İ汾��
run.sh 						run a set of different paramater experiments
#####����һ�鲻ͬ�Ĳ���ʵ��
doc.sh   					update sphinx documentation that locates in ../docs
#####����λ��./docs�е�sphinx documentation


## Explanation of Configurations:
#####���ý���˵��

### Network:

  --vocab_dim VOCAB_DIM

  --char_emb_dim CHAR_EMB_DIM

  --char_hidden_dim CHAR_HIDDEN_DIM

  --sentence_hidden_dim SENTENCE_HIDDEN_DIM

  --latent_dim LATENT_DIM

  --tone_dim TONE_DIM

  --vowel_dim VOWEL_DIM



### Training/Testing:

  --is_train IS_TRAIN

  --optimizer OPTIMIZER

  --batch_size BATCH_SIZE

  --validate_size VALIDATE_SIZE

  --lr LR

  --beta1 BETA1

  --beta2 BETA2

  --weight_decay WEIGHT_DECAY

  --clip_norm CLIP_NORM

  --sample_times SAMPLE_TIMES

  --using_first_sentence USING_FIRST_SENTENCE

  --kl_min KL_MIN

  --kl_annealing KL_ANNEALING

  --char_dropout CHAR_DROPOUT

  --max_step MAX_STEP

  --max_epochs MAX_EPOCHS

  --is_vowel_tone {NONE,ALL,YUNJIAO}

  --vowel_tone_input {NONE,DEC_RNN,PROJECTION,VOWEL_GRU_TONE_PROJECTION}

  --vowel_type {CURRENT,NEXT}

  --tone_type {CURRENT,NEXT}

  --train_same_as_test TRAIN_SAME_AS_TEST

  --is_attention IS_ATTENTION

  --is_dec_embedding IS_DEC_EMBEDDING

  --is_guided IS_GUIDED

  --is_fixed_p IS_FIXED_P //this is using a fixed network(distribution) for prior network

  --is_tone_using_rule IS_TONE_USING_RULE // this is whether using referenced tone or implied tone from rule



### Misc:

  --is_gpu IS_GPU

  --load_path LOAD_PATH

  --log_step LOG_STEP

  --save_step SAVE_STEP

  --num_log_samples NUM_LOG_SAMPLES

  --log_level {INFO,DEBUG,WARN}

  --log_dir LOG_DIR

  --model_dir MODEL_DIR

  --data_dir DATA_DIR



## Instruction of Usage:

Most training and testing tasks are required to configure the config.py and main.py.  Few of them may need extra modifications on the trainer.py file.
#####�����ѵ���Ͳ���������Ҫ����config.py��main.py�������к�����Ҫ��train.py�ļ����ж�����޸ġ�
The model can be trained with and without metrical structure, and each setting we did 2 main experiments which are blind and guided(see the paper for the definition) . For both blind and guided experiments,  we tried 2 different approaches in order to show the semantic diversity and they are 'same z for different Cipai' and 'different z for same cipai'.
#####��ģ�Ϳ�����û�����ɽṹ������½���ѵ����ÿһ���������Ƕ�������������Ҫ��äĿ�Ժ͵����Ե�ʵ��(�������еĶ���)����������ʵ����˵ Ϊ����ʾ��������ԣ����ǳ��������ֲ�ͬ�ķ��������Ƕ��ڲ�ͬ��Cipai����ͬ��z��������ͬ��cipai�ǲ�ͬ��z��
Below is the tree structure of the code folder. There are 3 main folders, base folder, examples folder and trained_parameters. Each training, testing and experiment's variations can be found in examples folder and can be executed by pasting to the base folder.
#####�����Ǵ����ļ��е����ṹ�����������ļ��С������ļ��С�ʾ���ļ��к�ѵ��������ÿ��ѵ�������Ժ�ʵ��ı仯��������ʾ���ļ����ҵ������ҿ���ͨ��ճ�������ļ�����ִ�С�
 The trained_parameters folder is the saved parameter and can be executed by pasting to the base folder.
#####ѵ�������ļ����Ǳ���Ĳ���������ͨ��ճ�������ļ�����ִ�С�
The input data used for training and test is in the folder 'base/Data' .
#####����ѵ���Ͳ��Ե���������λ�ڡ�base/data���ļ����С�
The version of the Pytorch is 0.4.0a0+6dc1fc7.
#####PYTORCH�İ汾Ϊ0.4.0a06dc1fc7��
.
|____base
| |____config.py
| |____Data
| | |____.gitignore
| | |____ci_per_line.txt
| | |____ci_per_line_after_cut.txt
| | |____cut_all.json
| | |____num.json
| | |____rules.json
| | |____sim.json
| | |____temp.py
| | |____title_lib.txt
| | |____tone.json
| | |____tone.txt
| | |____vocab.txt
| | |____vowel.json
| | |____vowel.txt
| |____debug.py
| |____doc.sh
| |____evaluation.py
| |____eval_example.py
| |____ioHelper.py
| |____iterator.py
| |____main.py
| |____models.py
| |____rules.py
| |____run.sh
| |____show.py
| |____statistics_hardcoded.py
| |____temp.py
| |____tobe.py
| |____toneVowelHelper.py
| |____trainer.py
| |____userHelper.py
| |____utils.py
| |____version.sh
| |____word_seg.py
|____documentation.md
|____examples
| |____testing
| | |____different semantic different z
| | | |____config.py
| | | |____main.py
| | |____different z for same cipai
| | | |____config.py
| | | |____main.py
| | |____Metrical_Performances
| | | |____config.py
| | | |____main.py
| | |____same z for different Cipai
| | | |____config.py
| | | |____main.py
| |____training
| | |____blind
| | | |____config.py
| | | |____main.py
| | |____guided
| | | |____config.py
| | | |____main.py
|____trained_paramaters
| |____config.py
| |____main.py
| |____saved_models
| | |____.DS_Store
| | |____100_128_200_24_0.500_ALL_DEC_RNN_NEXT_NEXT_is_dec_embedding
| | | |____.DS_Store
| | | |____Vrae_4_10239.pth
