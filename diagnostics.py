from datetime import datetime
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_curve

import numpy as np
import pickle 
import matplotlib.pyplot as plt
import os
import itertools
import seaborn as sn  

CLASS_NAMES = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', \
               'identity_hate']  

class Diagnostics:
	"""
	Encapsulates all our diagnostics so we can import to our model scripts and keep track of
	important stats
	Args:
	build: string (tf or sklearn) indicating whether model was generated using tf or tensorflow
	model_type: string indicating type of model (i.e. logistic, RNN, RF)
	preds_targets:nX12 matrix of prediction probabilities and target labels
	to use:  
		diag = Diagnostics(build='tf', model_type='logistic', preds_targets=pred_mat), example in logistic_baseline_tensorflow.py
		diag.do_all_diagnostics()
	"""
	def __init__(self,build, model_type, preds_targets, dataset):
		self.build = build #sklearn or tensorflow
		self.model_type = model_type
		self.preds_targets = preds_targets 
        self.dataset = dataset
		
		(ts, micro) = datetime.utcnow().strftime('%Y%m%d%H%M%S.%f').split('.')
		ts = "%s%03d" % (ts, int(micro) / 1000)
		self.output_dir = 'model_info/' + dataset + "/" + str(build) + "/" + str(model_type) + "/" + ts + "/"
		
		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir) 

	

	def get_confusion_matrices(self, cutoff_threshold = 0.5, save=True):
		"""
		Args:
		  cutoff_threshold: value between 0 and 1 used to cutoff positive from negative examples
		Returns:
		  raw confusion matrices as well as pretty plot, 2 x 3 with one confusion matrix for each
		  independent model
		"""
		preds = self.preds_targets[:,0:6]
		targets = self.preds_targets[:,6:12]
		classifications = np.where(preds > cutoff_threshold, 1, 0)
		
		fig, ax_array = plt.subplots(2,3)
		ax_array = ax_array.ravel()

		cm = []
		
		for i, ax_row in enumerate(ax_array):
			classification = classifications[:,i]
			target = targets[:,i]
			
			curr_cm = confusion_matrix(target, classification)
			cm.append(curr_cm)
			
			norm_cm = curr_cm.astype('float') / curr_cm.sum(axis=1)[:, np.newaxis]
			sn.heatmap(norm_cm,annot=True,annot_kws={"size": 16}, ax=ax_row)
			ax_row.set_title(CLASS_NAMES[i])

		if save:
			plot = self.output_dir + 'confusion_matrix_plots.png'
			fig.savefig(plot)

			raw_cm_fn = self.output_dir + 'confusion_matrices.pkl'
			pickle.dump(cm, open(raw_cm_fn, "wb" ) )




		plt.show()


	def get_roc_plots(self, savefig = True):
		"""
		returns roc plots for each class
		TODO: ADD AUC to legend
		"""
		preds = self.preds_targets[:,0:6]
		targets = self.preds_targets[:,6:12]

		fig, ax_array = plt.subplots(2,3)
		ax_array = ax_array.ravel()
		
		for i, ax_row in enumerate(ax_array):
			
			fpr, tpr, threshold = roc_curve(targets[:,i], preds[:,i])
			ax_row.plot(fpr, tpr, color='darkorange', lw=2)
			ax_row.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
			ax_row.set_title(CLASS_NAMES[i])
			
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
		
		if savefig:
			fn = self.output_dir + 'roc_plots.png'
			fig.savefig(fn)

		plt.show()




	def make_pred_prob_histograms(self, savefig = True):
		"""
		for each of 6 classes, overlay probabilities of negative examples with probabilites of 
		positive examples

		TODO: probably want to separate the positive and negative examples since there
		are so few positive examples which makes it hard to read
		"""
		preds = self.preds_targets[:,0:6]
		targets = self.preds_targets[:,6:12]

		pos_examples = preds[targets==1]
		neg_examples = preds[targets==0]

		fig, ax_array = plt.subplots(2,3)
		ax_array = ax_array.ravel()
		
		for i, ax_row in enumerate(ax_array):
			pos_col = preds[:,i]
			pos_examples = pos_col[targets[:,i]==1]
			
			neg_col = preds[:,i]
			neg_examples = neg_col[targets[:,i]==0]
			
			ax_row.hist(pos_examples)
			ax_row.hist(neg_examples)
			ax_row.set_title(CLASS_NAMES[i])

		if savefig:
			fn = self.output_dir + 'pred_prob_hist.png'
			fig.savefig(fn)

		plt.show()

	def do_all_diagnostics(self):
		self.get_confusion_matrices()
		self.get_roc_plots()
		self.make_pred_prob_histograms()

		

if __name__ == "__main__": 
	with open('pred_mat.pickle') as f:
		pred_mat = pickle.load(f)
	diagnostics = Diagnostics(build = 'tf', model_type = 'logistic-zero', preds_targets = pred_mat)
	diagnostics.get_confusion_matrices()
	









