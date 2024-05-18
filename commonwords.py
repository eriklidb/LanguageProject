if __name__ == '__main__':
    path1 = 'data_nytimes_strat5/val.txt__vocab.txt'
    path2 = 'data_nytimes_strat5/train.txt__vocab.txt'

    counts = {}

    with open(path1, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line not in counts:
                counts[line] = 0
            counts[line] += 1
    
    with open(path2, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if line not in counts:
                counts[line] = 0
            counts[line] += 1

    for line in counts:
        if counts[line] >= 2:
            print(line, end='')
