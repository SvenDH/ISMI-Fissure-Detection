import pandas as pd
import time

from utils import BatchGenerator, dice_coefficient, dice_coefficient_loss #custom file for utilities
from callbacks import SlackLogger #custom file for callbacks
from models import unet

from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import RandomOverSampler

# Loading data from pickle:
data = pd.read_pickle("train-data-filelist.pkl")

splitter = StratifiedShuffleSplit(1, test_size=0.1)

for train_index, test_index in splitter.split(data, data['label'].values):
    train_set = data.loc[train_index]
    validation_set = data.loc[test_index]

patch_size = (132,132,116) # smallest possible patch size is (108,108,108)
batch_size = 6 # 16 is max due to gpu memory errors

sampler = RandomOverSampler(random_state=42)

train_generator = BatchGenerator(train_set, patch_size, batch_size=batch_size, sampling=sampler.fit_sample)
validation_generator = BatchGenerator(validation_set, patch_size, batch_size=batch_size)


if __name__ == "__main__":
    model = unet(input_shape=[*patch_size,1])

    timeNow = time.strftime("%e%m-%H%M%S")

    slacklogger = SlackLogger()
    tensorboard = TensorBoard(log_dir='./logs/log-'+str(timeNow), batch_size=batch_size)
    modelcheck = ModelCheckpoint("weights-"+str(timeNow)+".hdf5", monitor='val_loss', verbose=0, save_best_only=True,
                                 save_weights_only=False, mode='auto', period=1)

    model.fit_generator(generator=train_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=10000,
                        validation_steps=1000,
                        epochs=10,
                        callbacks=[modelcheck, slacklogger, tensorboard],
                        verbose=1)

