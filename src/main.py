
import pandas as pd
from utils import *

if __name__ == '__main__':
    train_file = './data/cm_train.csv'
    test_file = './data/cm_test.csv'
    model_file = './model/bert.model'
    n_records_train = 1000
    n_records_test = 100
    batch_size = 32
    n_epochs = 10
    max_len = 256

    if n_records_train != -1:
        df = pd.read_csv('{}'.format(train_file), delimiter=',', nrows=n_records_train)
    else:
        df = pd.read_csv('{}'.format(train_file), delimiter=',')

    if n_records_test != -1:
        dft = pd.read_csv('{}'.format(test_file), delimiter=',', nrows=n_records_test)
    else:
        dft = pd.read_csv('{}'.format(test_file), delimiter=',')

    df = df[df['input'].str.len() <= max_len]
    dft = dft[dft['input'].str.len() <= max_len]

    train(df, n_epochs=n_epochs, batch_size=batch_size, max_len=max_len, model_file=model_file, show_loss=False, verbose=True)

    #dfa = binary_word_swap_attack_df(df)

    #print(character_level_attack_df(df, fraction_changed=0.15)

    evaluate(dft, max_len=max_len, batch_size=batch_size, model_file=model_file, verbose=True)

    #character_level_attack_df(dft)
