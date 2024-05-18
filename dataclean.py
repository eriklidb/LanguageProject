import argparse


def clean(samples_path, out_path, min_freq):
    vocab = set()
    counts = {}
    samples = []

    with open(samples_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            samples.append(line)
            label = line.strip().split(',')[1]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
            if label == 'presidencyadministration':
                print('Weird!', label)
                print('count:', counts[label])
            if counts[label] >= min_freq:
                vocab.add(label)

    with open(out_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            label = sample.strip().split(',')[1]
            if label in vocab:
                f.writelines([sample])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clean a samples file by removing uncommon words.', usage='\n* -d Samples file path. -n Minimum frequency for a label to be included. -o File to write cleaned samples to.')
    parser.add_argument('-d', type=str, help='Samples file path')
    parser.add_argument('-o', type=str, help='Output samples file path')
    parser.add_argument('-n', type=int, help='Minimum frequency for a label to be included')
    arguments = parser.parse_args()
    samples_path = arguments.d
    out_path = arguments.o
    min_freq = arguments.n

    clean(samples_path, out_path, min_freq)


