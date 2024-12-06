import os
import random
import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
import Data.DataEncoding.encodingList as encoding_list
from Models.ModelList import CRISPR_Net, cnn_std, CRISPR_IP, CRISPR_DNT, CnnCrispr, CRISPR_OFFT
from Models.mymodel import CRISPR-MCA


SEED = 2023
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def load_data(file_path):
    return pd.read_csv(file_path, header=1)


def predict_and_save(dataset_name, MODEL_NAME, y_true, y_score, fold, TAG):
    result_dir = f"./Result/ModelData/Extension_new/{MODEL_NAME}"
    os.makedirs(result_dir, exist_ok=True)

    model_dir = f"{result_dir}/{dataset_name}"
    os.makedirs(model_dir, exist_ok=True)

    result_df = pd.DataFrame({'Label': y_true, 'Score': y_score})
    result_df.to_csv(f"{model_dir}/fold_{fold}-{TAG}.csv", index=False)


def log_results(results, model_name):
    log_dir = "./Result/Log/Extension_new"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, f"{model_name}_log.txt")
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_path, 'a') as log_file:
        log_file.write(f"\n{current_time} {model_name}\n")
        log_file.write(results.to_string(index=False))
        log_file.write("\n\n")


def set_callbacks():
    early_stopping = EarlyStopping(
        monitor='val_loss', min_delta=0.0001, patience=10, verbose=0, mode='auto'
    )
    return [early_stopping]


def save_model(dataset_name,model, model_name, fold, tag):
    model_dir = f"./Result/Model/Extension_new/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    model_dir = f"{model_dir}/{dataset_name}"
    os.makedirs(model_dir, exist_ok=True)

    model.save(f'{model_dir}/{model_name}-{fold}-{tag}.h5')


def load_and_extend_data(x,y,dataname,xtest, ytest, tag):
    print("loding extention ModelData")
    positive_indices = np.where(ytest[:] == 1)[0]
    positive_samples_xtest = xtest[positive_indices]
    exdatafile = f"./Data/DataSets/extension/{dataname}_{tag}.csv"
    print(exdatafile)
    extensionData = np.array(pd.read_csv(exdatafile, header=0))
    exdata = extensionData[:, 0:2]
    keep_indices = []
    score = 0.0

    for i in range(0, len(extensionData), 16):
        origin_row = extensionData[i]
        origin_data = extensionData[i, 0:2]
        origin_score = float(origin_row[2])
        origin_spec = float(origin_row[3])
        origin_value = origin_score + origin_spec

        if not any(np.array_equal(origin_data, row) for row in positive_samples_xtest):
            for j in range(i + 1, i + 16):
                if j >= len(extensionData):
                    continue
                extension_row = extensionData[j]
                extension_score = float(extension_row[2])
                extension_spec = float(extension_row[3])
                extension_value = extension_score + extension_spec
                if extension_value > origin_value:
                    keep_indices.append(j)

    exdata = exdata[keep_indices]
    eydata = np.ones_like(exdata[:,0], dtype=int)
    xtrain = np.concatenate([x, exdata], axis=0)
    ytrain = np.concatenate([y, eydata], axis=0)

    return xtrain,ytrain

def plot_training_history(all_history):
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    for idx, history in enumerate(all_history):
        axes[idx].plot(history.history['loss'], label='Training Loss', color='blue')
        axes[idx].plot(history.history['val_loss'], label='Validation Loss', color='red')
        axes[idx].set_title(f'Fold {idx+1}')
        axes[idx].set_xlabel('Epochs')
        axes[idx].set_ylabel('Loss')
        axes[idx].legend()
    plt.tight_layout()
    plt.show()

def run():
    NUM_CLASSES = 2
    EPOCHS = 500
    BATCH_SIZE = 256
    LEARNING_RATE = 0.001
    TAG = f'Extension' 
    SHOW_LOSS = 0

    param_combinations = [
        {
            'train': 0,
            'coding': "onehot",
            'encoding_function': encoding_list.crispr_net_coding,
            'model_function': CRISPR-MCA,
            'coding_size_l': 24,
            'coding_size_h': 7,
            'dataset_list': ["Listgarten"]
        }
        
    ]
    for params in param_combinations:
        if params['train'] == 0:
            continue

        ENCODING_FUNCTION = params['encoding_function']
        MODEL_FUNCTION = params['model_function']
        MODEL_NAME = MODEL_FUNCTION.__name__
        CODING_SIZE_L = params['coding_size_l']
        CODING_SIZE_H = params['coding_size_h']
        dataset_list = params['dataset_list']
        coding = params['coding']

        for dataset_name in dataset_list:
            results = pd.DataFrame(columns=[
                'Fold', 'Accuracy', 'F1_score', 'Precision', 'Recall', 'ROC_AUC', 'PR_AUC'
            ])
            print(f"starting train {dataset_name} on {MODEL_NAME}")
            print('staring Encoding!!')

            encoded_file = f"./Data/DataSets/Mismatch/{dataset_name}.csv"
            train_data = np.array(load_data(encoded_file))
            txdata = train_data[:, 0:2]
            tydata = train_data[:, 2]
            tydata = tydata.astype('int')
            txdata = np.array(pd.DataFrame(txdata))

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
            callbacks = set_callbacks()
            fold = 1
            all_history = []
            for train_index, test_index in skf.split(txdata, tydata):
                print(f"statingon {MODEL_NAME} : {dataset_name}:fold {fold}")
                xtrain, xtest, ytrain, ytest = txdata[train_index], txdata[test_index], tydata[train_index], tydata[test_index]
                Exclusionx = xtest
                Exclusiony = ytest
                if isinstance(xtest, np.ndarray):
                    xtest = pd.DataFrame(xtest)
                print(f"testdata encoding")
                xtest = np.array(
                    xtest.apply(lambda row: ENCODING_FUNCTION(row[0], row[1]), axis=1).to_list()
                )
                if coding == 'onehot':
                    xtest = xtest.reshape((len(xtest), 1, CODING_SIZE_L, CODING_SIZE_H))
                ytest = to_categorical(ytest, NUM_CLASSES)

                if "Extension" in TAG:
                    print("Begin rebalancing dataset")
                    positive_indices = np.where(ytrain[:] == 1)[0]
                    print(len(positive_indices))
                    xtrain, ytrain = load_and_extend_data(xtrain, ytrain, dataset_name, Exclusionx, Exclusiony, TAG)
                    xtrain, ytrain = shuffle(xtrain, ytrain, random_state=SEED)
                    positive_indices = np.where(ytrain[:] == 1)[0]
                    print(f"to :{len(positive_indices)}")

                if isinstance(xtrain, np.ndarray):
                    xtrain = pd.DataFrame(xtrain)


                print(f"tarindata encoding")
                train_data_encodings = np.array(
                    xtrain.apply(lambda row: ENCODING_FUNCTION(row[0], row[1]), axis=1).to_list()
                )
                if coding == 'onehot':
                    train_data_encodings = train_data_encodings.reshape((len(train_data_encodings), 1, CODING_SIZE_L, CODING_SIZE_H))
                train_labels = to_categorical(ytrain, NUM_CLASSES)

                model = MODEL_FUNCTION()

                model.compile(optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE, amsgrad=False), loss=tf.keras.losses.CategoricalCrossentropy(),
                              metrics=['accuracy', 'Precision', 'Recall'])

                train_data_encodings = train_data_encodings.astype('float32')
                xtest = xtest.astype('float32')
                xval = xval.astype('float32')
                train_labels = train_labels.astype('float32')
                ytest = ytest.astype('float32')
                yval = yval.astype('float32')

                history = model.fit(
                    train_data_encodings, train_labels,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    callbacks=callbacks,
                    validation_split=0.2
                )

                save_model(dataset_name, model, MODEL_NAME, fold, TAG)

                all_history.append(history)
                pre = model.predict(xtest)
                y_pred = np.argmax(pre, axis=1)
                y_true = np.argmax(ytest, axis=1)
                y_score = pre[:, 1]

                predict_and_save(dataset_name, MODEL_NAME, y_true, y_score, fold, TAG)

                results_dict = {'Fold': fold}
                metrics = {'Accuracy': accuracy_score, 'F1_score': f1_score, 'Precision': precision_score,
                           'Recall': recall_score,
                           'ROC_AUC': roc_auc_score, 'PR_AUC': average_precision_score}
                for metric_name, metric_func in metrics.items():
                    score = metric_func(y_true, y_pred) if 'AUC' not in metric_name else metric_func(y_true, y_score)
                    results_dict[metric_name] = round(score, 4)

                results = results.append(results_dict, ignore_index=True)
                fold += 1
            metrics = ['Accuracy', 'F1_score', 'Precision', 'Recall', 'ROC_AUC', 'PR_AUC']
            recent_results = results.tail(5)
            average_row = {'Fold': 'Average'}
            for metric in metrics:
                average_row[metric] = round(recent_results[metric].mean(), 4)
            results = results.append(average_row, ignore_index=True)
            results = results.append(
                {'Fold': f"{MODEL_NAME}", "Accuracy": f'{dataset_name}', "F1_score": f'ep:{EPOCHS}',
                 "Precision": f'bz:{BATCH_SIZE}',
                 "Recall": f'lr:{LEARNING_RATE}', "ROC_AUC": f'', "PR_AUC": f'{TAG}'},
                ignore_index=True)

            log_results(results, MODEL_NAME)
            print(results)

if __name__ == '__main__':
    run()
