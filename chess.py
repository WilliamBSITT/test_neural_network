#!/usr/bin/env python3
import os
import glob
import numpy as np
from loguru import logger
import json
import sys
from nn.model import NeuralNetwork
from nn.callback import EarlyStopping, ModelCheckpoint, CSVLogger
from compress_nn import compress_model_object

# Mapping for pieces to indices in the 12-channel vector
PIECE_MAP = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

LABEL_MAP = {
    'Nothing': 0,
    'Check White': 1,
    'Check Black': 2,
    'Checkmate White': 3,
    'Checkmate Black': 4
}

INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

def parse_fen(fen):
    parts = fen.split()
    board_str = parts[0]
    turn = parts[1]
    
    # 8x8 board, 12 channels for pieces
    board = np.zeros((64, 12), dtype=np.float32)
    
    ranks = board_str.split('/')
    square_idx = 0
    
    for rank in ranks:
        for char in rank:
            if char.isdigit():
                square_idx += int(char)
            else:
                if char in PIECE_MAP:
                    board[square_idx, PIECE_MAP[char]] = 1.0
                square_idx += 1
                
    # Flatten the board
    board_flat = board.flatten()
    
    # Add turn info (1 for white, 0 for black)
    turn_val = 1.0 if turn == 'w' else 0.0
    
    return np.concatenate([board_flat, [turn_val]])

def load_data(dataset_dir):
    # logger.info(f"Loading data from {dataset_dir}")
    files = glob.glob(os.path.join(dataset_dir, "*.txt"))
    
    x_data = []
    y_data = []
    
    for file_path in files:
        # logger.info(f"Processing {file_path}")
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # Split FEN and Label
                # FEN has 6 parts separated by space. Label follows.
                # Example: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing
                # But some labels have spaces: Check White
                
                parts = line.split()
                # FEN is first 6 parts
                fen = " ".join(parts[:6])
                label_str = " ".join(parts[6:])
                
                if label_str not in LABEL_MAP:
                    continue
                    
                x = parse_fen(fen)
                y = np.zeros(len(LABEL_MAP), dtype=np.float32)
                y[LABEL_MAP[label_str]] = 1.0
                
                x_data.append(x)
                y_data.append(y)
        
    return np.array(x_data), np.array(y_data)

def train_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    epochs=50,
    batch_size=64,
    callbacks=(),
    initial_epoch=0,
):
    num_samples = x_train.shape[1]
    num_batches = int(np.ceil(num_samples / batch_size))

    # attach model to callbacks
    for cb in callbacks:
        try:
            cb.model = model
        except Exception:
            pass

    start = initial_epoch + 1
    end = initial_epoch + epochs

    for epoch in range(start, end + 1):
        # Shuffle
        indices = np.random.permutation(num_samples)
        x_shuffled = x_train[:, indices]
        y_shuffled = y_train[:, indices]

        epoch_loss = 0

        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            batch_x = x_shuffled[:, start_idx:end_idx]
            batch_y = y_shuffled[:, start_idx:end_idx]

            # manual forward/backward/update per batch
            model._input = batch_x
            _ = model(model._input)
            batch_loss = model._loss(model._output, batch_y)
            model.backward_step(batch_y)
            model.update()

            epoch_loss += float(np.squeeze(batch_loss))

        avg_loss = epoch_loss / num_batches

        # validation loss
        val_loss = float(np.squeeze(model.evaluate(x_val, y_val)))

        # notify callbacks with validation loss
        for cb in callbacks:
            cb.on_epoch_end(epoch, val_loss)

        logger.info(f"Epoch {epoch}/{end}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}")

        # check stop
        if any(getattr(cb, 'stop_training', False) for cb in callbacks):
            break

def run_predict(loadfile, chessfile):
    if os.path.exists(loadfile):
        # logger.info(f"Loading existing model from {loadfile}")
        model = NeuralNetwork.load(loadfile)
    else:
        sys.exit(84)

    with open(chessfile, 'r') as chessfile:
        data = [line.strip() for line in chessfile if line.strip()]
    chessfile.close()

    for fen in data:
        x_val = parse_fen(fen)
        # Reshape for prediction (769, 1)
        x_val = x_val.reshape(-1, 1)
        pred = model.predict(x_val)
        pred_class = np.argmax(pred, axis=0)[0]
        label = INV_LABEL_MAP[pred_class]
        # logger.info(f"FEN: {fen}")
        # logger.info(f"Prediction: {label}")
        print(label)
    pass

def run_train(loadfile, chessfile, savefile=None, epochs=200, batch_size=64):

    # Check if dataset exists
    if not os.path.isdir(chessfile):
        # logger.error(f"Dataset directory {chessfile} not found.")
        return

    # Load data
    x, y = load_data(chessfile)
    
    # logger.info(f"Loaded {len(x)} samples.")

    x = x.T
    y = y.T
    
    # Split train/test
    num_samples = x.shape[1]
    indices = np.random.permutation(num_samples)
    split_idx = int(num_samples * 0.8)
    
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    x_train, y_train = x[:, train_indices], y[:, train_indices]
    x_test, y_test = x[:, test_indices], y[:, test_indices]
    
    if os.path.exists(loadfile):
        logger.info(f"Loading existing model from {loadfile}")
        model = NeuralNetwork.load(loadfile)
    else:
        sys.exit(84)
    
    # logger.info("Training model")

    # Prepare callbacks: checkpoint + early stopping
    checkpoint = ModelCheckpoint(loadfile, save_best_only=True)
    early = EarlyStopping(patience=15)
    csvlogger = CSVLogger(loadfile + '.log', overwrite=False) if not os.path.exists(loadfile + '.log') else None
    callbacks = [checkpoint, early]
    if csvlogger:
        callbacks.append(csvlogger)

    # resume support: read meta file if exists
    meta_path = loadfile + '.meta'
    last_epoch = 0
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as mf:
                meta = json.load(mf)
                last_epoch = int(meta.get('last_epoch', 0))
                logger.info(f"Resuming from epoch {last_epoch}")
        except Exception:
            last_epoch = 0

    epochs_to_run = max(0, epochs - last_epoch)

    if epochs_to_run > 0:
        train_model(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            epochs=epochs_to_run,
            batch_size=batch_size,
            callbacks=callbacks,
            initial_epoch=last_epoch,
        )
    
    
    print("Compressing model before saving...")
    compress_model_object(model)
    
    # logger.info(f"Saving model to {loadfile}")
    if savefile is not None:
        model.save(savefile)
    else:
        model.save(loadfile)

    
    #### Allow to evaluate after training if needed

    # logger.info("Evaluating model")
    # loss = model.evaluate(x_test, y_test)
    # logger.info(f"Validation loss: {np.squeeze(loss):.4f}")
    
    # preds = model.predict(x_test)
    
    # # Accuracy
    # # preds is (5, num_samples)
    # pred_classes = np.argmax(preds, axis=0)
    # true_classes = np.argmax(y_test, axis=0)
    # acc = np.mean(pred_classes == true_classes)
    # logger.info(f"Test accuracy: {acc:.4f}")

if __name__ == "__main__":
    argv = sys.argv[1:]
    if not argv or any(a in ('-h', '--help') for a in argv):
        sys.exit(0)

    mode = None
    savefile = None
    i = 0
    # parse options
    while i < len(argv):
        a = argv[i]
        if a in ('--train', '-t'):
            if mode is not None:
                print('Error: only one of --train or --predict may be specified', file=sys.stderr)
                sys.exit(84)
            mode = 'train'
            i += 1
        elif a in ('--predict', '-p'):
            if mode is not None:
                print('Error: only one of --train or --predict may be specified', file=sys.stderr)
                sys.exit(84)
            mode = 'predict'
            i += 1
        elif a == '--save':
            if i + 1 >= len(argv):
                print('Error: --save requires a filename', file=sys.stderr)
                sys.exit(84)
            savefile = argv[i + 1]
            i += 2
        elif a.startswith('-'):
            print(f'Unknown option: {a}', file=sys.stderr)
            sys.exit(84)
        else:
            break

    pos = argv[i:]
    if mode is None or len(pos) != 2:
        sys.exit(84)

    loadfile, chessfile = pos[0], pos[1]
    # print(f'Loadfile: {loadfile}, Chessfile: {chessfile}, Mode: {mode}, Savefile: {savefile}')

    if mode == 'train':
        run_train(loadfile, chessfile, savefile=savefile, epochs=200, batch_size=64)
    elif mode == 'predict':
        if savefile is not None:
            print('Warning: --save ignored in predict mode', file=sys.stderr)
            sys.exit(84)
        run_predict(loadfile, chessfile)
    else:
        sys.exit(84)
