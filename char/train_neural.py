import argparse
from data import DataLoader
from neural import NeuralPredictor


def main(model_path, data_path, epochs):
    if None in [model_path, data_path, epochs]:
        print('Invalid arguments')
        exit(-1)
    
    data_src = DataLoader(data_path)
    print('Data source constructed')
    model = NeuralPredictor(data_src, 3, epochs=epochs)
    print("Saving...")
    model.save(model_path)
    print("Done saving.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train neural model.', usage='\n* -m Model file path. -d Data filepath. -e Number of epochs to run.')
    parser.add_argument('-m', type=str, help='Directory to save model to')
    parser.add_argument('-d', type=str, help='Data filepath (e.g. train.txt)')
    parser.add_argument('-e', type=int, help='Number of epochs to run.')
    arguments = parser.parse_args()
    
    model_path = arguments.m
    data_path = arguments.d
    epochs = arguments.e
    main(model_path, data_path, epochs)


