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
	def __init__(self, validation_data, patch_size, stride=1):
		self.val_imgs = pad(validation_data.imgs, patch_size) / 255
		self.val_lbls = downscale(validation_data.lbls, stride) > 0
		self.val_msks = downscale(validation_data.msks, stride) > 0
		 
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

		dice = self.validate()
		self.dices.append([len(self.losses), dice])
		if dice > self.best_dice:
			self.best_dice = dice
			self.best_model = self.model.get_weights()
		self.plot()

	def on_train_end(self, logs={}):

		sendSlack("-------- Results of Training ---------")
		sendSlack("Average time per epoch (secs): " + str(np.mean(self.times)))
		sendSlack("Best dice score: " + str(self.beste_dice))
		sendSlack("Last loss: " + str(self.loss[len(self.loss)-1]))
		sendSlack("Average loss: " + str(np.mean(self.loss)))
		

	def validate(self):
		predicted_lbls = self.model.predict(self.val_imgs, batch_size=1)[:,:,:,1]>0.5
		x = self.val_lbls[self.val_msks]
		y = predicted_lbls[self.val_msks]
		return calculate_dice(x, y)
	
	def plot(self):
		clear_output()
		N = len(self.losses)
		train_loss_plt, = plt.plot(range(0, N), self.losses)
		dice_plt, = plt.plot(*np.array(self.dices).T)
		plt.legend((train_loss_plt, dice_plt), 
				   ('training loss', 'validation dice'))
		plt.show()
