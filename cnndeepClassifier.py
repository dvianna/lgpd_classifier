from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras import layers
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras import backend as K


def deep_classifier(documents, labels):
    doc_train, doc_test, label_train, label_test = train_test_split(documents, labels, test_size=0.25)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(doc_train)
    Xcnn_train = tokenizer.texts_to_sequences(doc_train)
    Xcnn_test = tokenizer.texts_to_sequences(doc_test)
    vocab_size = len(tokenizer.word_index) + 1

    maxlen = 2000
    Xcnn_train = pad_sequences(Xcnn_train, padding='post', maxlen=maxlen)
    Xcnn_test = pad_sequences(Xcnn_test, padding='post', maxlen=maxlen)

    metrics = Metrics()

    embedding_dim = 1000
    txtcnn = Sequential()
    txtcnn.add(layers.Embedding(vocab_size, embedding_dim, input_length=maxlen))
    txtcnn.add(layers.Conv1D(128, 5, activation='relu'))
    txtcnn.add(layers.GlobalMaxPooling1D())
    txtcnn.add(layers.Dense(10, activation='relu'))
    txtcnn.add(layers.Dense(1, activation='sigmoid'))
    txtcnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1])
    txtcnn.summary()

    txtcnn.fit(Xcnn_train,
               label_train,
               epochs=30,
               verbose=False,
               validation_data=(Xcnn_test, label_test),
               batch_size=24)
    loss, accuracy = txtcnn.evaluate(Xcnn_train, label_train, verbose=False)
    print('Training accuracy: {:.4f}'.format(accuracy))
    loss, accuracy = txtcnn.evaluate(Xcnn_test, label_test, verbose=False)
    print('Test accuracy: {:.4f}'.format(accuracy))


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
        val_targ = self.model.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        print (" — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall))
        return


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))