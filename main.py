from tensorflow.keras import Input
from model import build_model, train_model
from data import load_and_split_data


def main():
    data_dir = 'data2'
    path = 'data2/driving_log.csv'
    column_names = ['center', 'left', 'right', 'steer']

    im_train, im_val, im_train, *labels = load_and_split_data(
        path=path, data_dir=data_dir, column_names=column_names
    )
    labels_train, labels_val, labels_test = labels

    lr = 1.0e-4
    ip = Input(shape=(66, 200, 3))
    model = build_model(ip=ip, activation='elu', lr=lr)
    model.summary()

    train_model(
        model=model, im_train=im_train, labels_train=labels_train,
        im_val=im_val, labels_val=labels_val
    )


if __name__ == '__main__':
    main()
