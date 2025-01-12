all:
	echo $(ppl) > generation_ppl.txt
	cp hw4p1/experiments/$(runid)/prediction-probs-test-$(epoch).npy prediction_probs.npy
	cp hw4p1/experiments/$(runid)/generated-texts-$(epoch)-test.txt generated_texts.txt
	cp hw4p1/hw4p1.ipynb training.ipynb
	cp hw4p1/attention.py attention.py
	tar -cvzf handin.tar training.ipynb prediction_probs.npy generated_texts.txt generation_ppl.txt attention.py
	rm -f generated_texts.txt prediction_probs.npy training.ipynb generation_ppl.txt attention.py
