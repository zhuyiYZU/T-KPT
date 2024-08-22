This is the repository for the " Two-stage Knowledgeable Prompt-tuning for Chinese Implicit Hate Speech Detection".

First, by pip install -r requirement.txt to install all the dependencies.

Firstly install OpenPrompt https://github.com/thunlp/OpenPrompt

Then copy prompts/knowledgeable_verbalizer.py to Openprompt/openprompt/prompts/knowledgeable_verbalizer.py

And install BERT model https://huggingface.co/bert-chinese to models/

Also, you can put your own dataset in datasets/TextClassification.

example shell scripts:

python fewshot.py --result_file ./output_fewshot.txt --dataset proscons --template_id 0 --seed 123 --shot 20 --verbalizer ex_re1
