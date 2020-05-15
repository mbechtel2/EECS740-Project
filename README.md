# EECS740-Project

Steps to recreate report results:

1. Generate Spectrum images for Histogram Matching:

	$ python create_spectrums.py 320 240 4 # Figure 3

2. Create simulated copies of training, validation, and test data:
	```
	$ mkdir figs
	$ python match_video.py
	$ python powerlaw_video.py
	$ python gen_histograms.py   # Figure 1
	```
3. Train the CNN models:
	```
	$ mkdir models
	$ python train.py
	$ python train_gray.py
	$ python train_spec.py
	```
4. Evaluate the CNN models:
	```
	$ python eval_models.py       # Figure 2
	$ python eval_noge_models.py  # Figure 4
	$ python eval_spec_models.py  # Figure 5
	$ python eval_ge_models.py    # Figure 6
	```
5. Evaluate execution timings for each approach
	```
	$ python eval_prep_times.py   # Table 1
	```
