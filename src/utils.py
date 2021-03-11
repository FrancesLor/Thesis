import torch
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
import numpy as np
import random
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
#from PyDictionary import PyDictionary


device = torch.device("cuda")


def train(df, n_records_train = 50, n_epochs=4, batch_size=16, max_len=256, model_file='./model/bert.model', show_loss=False, verbose=False):
    # Load the dataset into a pandas dataframe.
    '''if n_records != -1:
        df = pd.read_csv('{}'.format(training_file), delimiter=',', nrows=n_records)
    else:
        df = pd.read_csv('{}'.format(training_file), delimiter=',')'''
    # Report the number of sentences.

    #Remove all entries with length longer than max_len
    #df = df[df['input'].str.len() <= max_len]

    # Get the lists of sentences and their labels.
    sentences = df.input.values
    labels = df.label.values

    # Tokenize the data using the BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, cache_dir = './model/tokenizer/')


    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sent = tokenizer.encode(
                            sent,
                            add_special_tokens = True,
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_sent)

    # Max sentence length is 2881, so we will use 512
    #MAX_LEN = 256
    if verbose:
        print('\nPadding/truncating all sentences to %d values...' % max_len)
        print('\nPadding token: "{:}", ID: {:}'.format(tokenizer.pad_token, tokenizer.pad_token_id))
    # Pad our input tokens with value 0.
    # "post" indicates that we want to pad and truncate at the end of the sequence,
    # as opposed to the beginning.
    input_ids = pad_sequences(input_ids, maxlen=max_len, dtype="long",
                              value=0, truncating="post", padding="post")
    if verbose:
        print('Done.')

    # Explicitly differentiate between token and padding
    # Create attention masks
    attention_masks = []
    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)

    # Split the training data into a training and validation set
    # Use train_test_split to split our data into train and validation sets for training
    from sklearn.model_selection import train_test_split
    # Use 90% for training and 10% for validation.
    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                random_state=2018, test_size=0.1)
    # Do the same for the masks.
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, labels,
                                                 random_state=2018, test_size=0.1)

    # Convert all inputs and labels into torch tensors, the required datatype
    # for our model.
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # CREATE AN ITERATOR TO SAVE ON MEMORY
    # The DataLoader needs to know our batch size for training, so we specify it
    # here.
    # For fine-tuning BERT on a specific task, the authors recommend a batch size of
    # 16 or 32.
    #batch_size = 32
    # Create the DataLoader for our training set.
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    # Create the DataLoader for our validation set.
    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)



    # Load BertForSequenceClassification, the pretrained BERT model with a single
    # linear classification layer on top.
    '''
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", # Use the 12-layer BERT model, with an uncased vocab.
        num_labels = 2, # The number of output labels--2 for binary classification.
                        # You can increase this for multi-class tasks.
        output_attentions = False, # Whether the model returns attentions weights.
        output_hidden_states = False, # Whether the model returns all hidden-states.
    )
    '''
    model = torch.load('./model/untrained.model')
    # Note: AdamW is a class from the huggingface library (as opposed to pytorch)
    # I believe the 'W' stands for 'Weight Decay fix"
    optimizer = AdamW(model.parameters(),
                      lr = 2e-5, # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps = 1e-8 # args.adam_epsilon  - default is 1e-8.
                    )

    # Number of training epochs (authors recommend between 2 and 4)
    #epochs = 4
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * n_epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0, # Default value in run_glue.py
                                                num_training_steps = total_steps)

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    # Set seed
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Store the average loss after each epoch so we can plot them.
    loss_values = []
    # For each epoch...
    for epoch_i in range(0, n_epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.
        if verbose:
            print("")
            print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, n_epochs))
            print('Training...')
    # Reset the total loss for this epoch.
        total_loss = 0
        model.train()
    # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            outputs = model(b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask,
                        labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
             # Perform a backward pass to calculate the gradients.
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
    # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)
        if verbose:
            print("")
            print("  Average training loss: {0:.2f}".format(avg_train_loss))
                # ========================================
            #               Validation
            # ========================================
            # After the completion of each training epoch, measure our performance on
            # our validation set.
            print("")
            print("Running Validation...")
        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()
        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        # Evaluate data for one epoch
        for batch in validation_dataloader:

            # Add batch to cpu
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # This will return the logits rather than the loss because we have
                # not provided labels.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                outputs = model(b_input_ids,
                                token_type_ids=None,
                                attention_mask=b_input_mask)

            # Get the "logits" output by the model. The "logits" are the output
            # values prior to applying an activation function like the softmax.
            logits = outputs[0]
            # Move logits and labels to cpu
            logits = logits.detach().cuda().numpy()
            label_ids = b_labels.to(device).numpy()

            # Calculate the accuracy for this batch of test sentences.
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            # Accumulate the total accuracy.
            eval_accuracy += tmp_eval_accuracy
            # Track the number of batches
            nb_eval_steps += 1
        # Report the final accuracy for this validation run.
        if verbose:
            print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))

    torch.save(model,'./model/bert.model')

    if verbose:
        print("")
        print("Training complete!")

    if show_loss:
        ### figure:  training loss over batches
        import plotly.express as px
        f = pd.DataFrame(loss_values)
        f.columns=['Loss']
        fig = px.line(f, x=f.index, y=f.Loss)
        fig.update_layout(title='Training loss of the Model',
                           xaxis_title='Epoch',
                           yaxis_title='Loss')
        fig.show()

def evaluate(dft, max_len=256, batch_size=16, model_file='./model/bert.model', verbose=False):
    ############## PREPARING TEST SET DATA  ######################
    # Report the number of sentences.
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    model = torch.load(model_file)
    if verbose:
        print('Number of test sentences: {:,}\n'.format(dft.shape[0]))
    # Create sentence and label lists
    sentencest = dft.input.values
    labelst = dft.label.values
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_idst = []
    # For every sentence...
    for sent in sentencest:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        encoded_sentt = tokenizer.encode(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                       )

        input_idst.append(encoded_sentt)
    # Pad our input tokens
    input_idst = pad_sequences(input_idst, maxlen=max_len,
                              dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_maskst = []
    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_idst:
      seq_maskt = [float(i>0) for i in seq]
      attention_maskst.append(seq_maskt)
    # Convert to tensors.
    prediction_inputs = torch.tensor(input_idst)
    prediction_masks = torch.tensor(attention_maskst)
    prediction_labels = torch.tensor(labelst)
    # Set the batch size.
    #batch_size = 32
    # Create the DataLoader.
    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

        # ========================================
        #           Predict on test set
        # ========================================
    if verbose:
        print('Predicting labels for {:,} test sentences...'.format(len(prediction_inputs)))
    # Put model in evaluation mode
    model.eval()
    # Tracking variables
    predictions , true_labels = [], []
    # Predict
    for batch in prediction_dataloader:
      # Add batch to cpu
      batch = tuple(t.to(device) for t in batch)

      # Unpack the inputs from our dataloader
      b_input_ids, b_input_mask, b_labels = batch

      # Telling the model not to compute or store gradients, saving memory and
      # speeding up prediction
      with torch.no_grad():
          # Forward pass, calculate logit predictions
          outputs = model(b_input_ids, token_type_ids=None,
                          attention_mask=b_input_mask)
      logits = outputs[0]
      # Move logits and labels to cpu
      logits = logits.detach().cuda().numpy()
      label_ids = b_labels.to('cuda').numpy()

      # Store predictions and true labels
      predictions.append(logits)
      true_labels.append(label_ids)
    if verbose:
        print('DONE.')

    # FINALLY ACCURACY
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    # Combine the correct labels for each batch into a single list.
    flat_true_labels = [item for sublist in true_labels for item in sublist]
    # Calculate the MCC
    acc = accuracy_score(flat_true_labels, flat_predictions)
    print('Accuracy: %.3f' % acc)
    cm  = confusion_matrix(flat_true_labels, flat_predictions)
    print(cm)

def character_level_attack(sentence, fraction_changed=0.05):
    PUNCTUATION = [' ', '.', ',', '\'', '!', '?', '\"', '-']
    ALPHA = list("abcedfghijklmnopqrstuvwxyz")
    changed_indices = []
    indices = np.random.choice(range(len(sentence)), int(len(sentence) * fraction_changed), replace=False)
    letters = np.random.choice(range(26), int(len(sentence) * fraction_changed), replace=True)

    sentence = list(sentence)
    for index, letter in zip(indices, letters):
        while sentence[index] in PUNCTUATION or index in changed_indices:
            index = np.random.randint(len(sentence))
        index = int(index)
        letter = int(letter)
        if ALPHA.index(sentence[index]) == letter:
            sentence[index] = ALPHA[(letter + 1) % 26]
        else:
            sentence[index] = ALPHA[letter]
        changed_indices.append(index)
    sentence = ''.join(sentence)
    return sentence

def character_level_attack_df(df, fraction_changed=0.15):
    for i, row in df.iterrows():
        df['input'][i] = character_level_attack(row['input'].lower())
    return df


def binary_word_swap_attack(sentence):
    PUNCTUATION = ['.', ',', '!', '?']
    sentences = sentence.split(' ')
    i1 = np.random.randint(len(sentences))
    if i1 == 0:
        i2 = 1
    else:
        i2 = i1 - 1

    word1 = sentences[i1]
    word1_puncindex = len(word1)
    word2 = sentences[i2]
    word2_puncindex = len(word2)

    for sym in PUNCTUATION:
        if sym in word1:
            if word1.index(sym) < word1_puncindex:
                word1_puncindex = word1.index(sym)
        if sym in word2:
            if word2.index(sym) < word2_puncindex:
                word2_puncindex = word2.index(sym)
    newword1 = word2[:word2_puncindex]+word1[word1_puncindex:]
    newword2 = word1[:word1_puncindex]+word2[word2_puncindex:]
    sentences[i1] =  newword1
    sentences[i2] = newword2

    sentence = ' '.join(sentences)

    return sentence

def binary_word_swap_attack_df(df):
    for i, row in df.iterrows():
        df['input'][i] = binary_word_swap_attack(row['input'].lower())
    return df

'''
def syn_attack(sentence):
    stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "dont", "should", "now"]
    PUNCTUATION = [' ', '.', ',', '\'', '!', '?', '\"', '-']
    sentence = sentence.lower()
    sentences = sentence.split(' ')
    i = np.random.randint(len(sentences))
    word = sentences[i]
    word_puncindex = len(word)

    for sym in PUNCTUATION:
        if sym in word:
            if word.index(sym) < word_puncindex:
                word_puncindex = word.index(sym)

    # check if the chosen word is a stop word. If it is, choose again.
    while word in stop_words:
        i = np.random.randint(len(sentences))
        word = sentences[i]
        word_puncindex = len(word)

        for sym in PUNCTUATION:
            if sym in word:
                if word.index(sym) < word_puncindex:
                    word_puncindex = word.index(sym)
    word = word[:word_puncindex]
    dictionary = PyDictionary()
    synonyms = dictionary.synonym(word)
    synword = synonyms[0]
    sentences[i] = synword + sentences[i][word_puncindex:]
    sentence = " ".join(sentences)
    return  sentence

def syn_attack_df(df):
    for i, row in df.iterrows():
        df['input'][i] = syn_attack(row['input'])
    return df
    '''
