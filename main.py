from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model import build_model, train_model
from data import flatten_csv


def compile_model(ip=Input(shape=(128, 128, 3)), lr=1e-3, activation='relu'):
    model = build_model(ip=ip, activation=activation)
    model.compile(loss='mse',
                  optimizer=Adam(lr=lr))
    return model


def load_and_split_data(path,
                        data_dir,
                        column_names,
                        test_size=.15,
                        val_size=.15):
    images, labels = flatten_csv(path=path,
                                 data_dir=data_dir,
                                 column_names=column_names)

    im_train, im_test, labels_train, labels_test = train_test_split(images,
                                                                    labels,
                                                                    test_size=test_size,
                                                                    shuffle=True)
    im_train, im_val, labels_train, labels_val = train_test_split(im_train,
                                                                  labels_train,
                                                                  test_size=val_size,
                                                                  shuffle=True)
    return im_train, im_val, im_train, labels_train, labels_val, labels_test


def plot_model_history(model):
    plt.plot(model.history.history['loss'], 'r', label='train')
    plt.plot(model.history.history['val_loss'], 'g', label='validation')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.legend()


def main():
    data_dir = 'data2'
    path = 'data2/driving_log.csv'

    im_train, im_val, im_train, labels_train, labels_val, labels_test = load_and_split_data(path=path, data_dir=data_dir, column_names=['center', 'left', 'right', 'steer'])

    lr = 1.0e-4
    ip = Input(shape=(66, 200, 3))

    model = compile_model(ip=ip, lr=lr)
    model.summary()

    train_model(model=model,
                im_train=im_train,
                labels_train=labels_train,
                im_val=im_val,
                labels_val=labels_val)

    plot_model_history(model=model)


if __name__ == '__main__':
    main()
