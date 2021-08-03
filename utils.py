import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, chi2
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout,
    Bidirectional,
    TimeDistributed,
    Flatten,
    BatchNormalization,
    Conv1D,
    MaxPooling1D,
    Activation,
    AveragePooling1D,
    RepeatVector,
    Conv2D,
    MaxPooling2D,
    Reshape,
)
from scipy import stats
from scipy.stats import chi2_contingency
import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras import Sequential
from image_creator import *


def data_read():
    df = pd.read_csv("data/data_male.csv")
    df_buffer = pd.DataFrame()
    df_buffer['thicc_forearm'] = df['forearmcenterofgriplength']
    df_buffer['thicc_forearm'] = df_buffer['thicc_forearm'].apply(lambda x: x+3)
    df_buffer['thicc_wrist'] = df['wristcircumference']
    df_buffer['thicc_wrist'] = df_buffer['thicc_wrist'].apply(lambda x: x+2)
    df_buffer['hand_lenght'] = df['handlength']
    df_buffer['hand_circunference'] = df['handcircumference'] 
    df_buffer['forearm_hand_lenght']  = df['forearmhandlength']
    print(df_buffer.head(4))
    A0 = np.array([1, 3, 2])
    A1 = np.array([14, 5, 9])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_zlim(0, 10)
    truncated_cone(A0, A1, 1, 5, 'blue')
    plt.show()
    return df_buffer