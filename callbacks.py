import json
import requests
import time
import numpy as np
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, TensorBoard


'''
Sends message to slack.
For convenience it also prints it.
'''
def sendSlack(message):
	print(message)
	dataT =  {'text': message}
	webhook = 'https://hooks.slack.com/services/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'
	response = requests.post(webhook, data=json.dumps(dataT), headers={'Content-Type':'application/json'})

'''
Copied the Logger that was used before I touched it. Added time per epoch and submits results to Slack
'''
class SlackLogger(Callback):
	def __init__(self):
		 
		self.losses = []
		self.dices = []
		self.best_dice = 0
		self.best_model = None

	def on_train_begin(self, logs={}):
		self.times = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
	
	def on_epoch_begin(self, batch, logs={}):
		self.epoch_time_start = time.time()

	def on_epoch_end(self, batch, logs={}):
		self.times.append(time.time() - self.epoch_time_start)


	def on_train_end(self, logs={}):

		sendSlack("-------- Results of Training ---------")
		sendSlack("Average time per epoch (secs): " + str(np.mean(self.times)))
		sendSlack("Last loss: " + str(self.losses[len(self.losses)-1]))
		sendSlack("Average loss: " + str(np.mean(self.losses)))
		
	