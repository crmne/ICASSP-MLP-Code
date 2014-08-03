import numpy
from numpy.random import RandomState
import os
import pickle
import utils as U
import argparse


def read_file(filename):
    """
    Loads a file into a list
    """
    file_list = [l.strip() for l in open(filename, 'r').readlines()]
    return file_list


def get_folds(filelist, n_folds):
    n_per_fold = len(filelist) / n_folds
    folds = []
    for i in range(n_folds - 1):
        folds.append(filelist[i * n_per_fold: (i + 1) * n_per_fold])
    i = n_folds - 1
    folds.append(filelist[i * n_per_fold:])
    return folds


def generate_mirex_list(train_list, annotations):
    out_list = []
    for song in train_list:
        annot = annotations.get(song, None)
        if annot is None:
            print 'No annotations for song %s' % song
            continue
        assert(type('') == type(annot))
        out_list.append('%s\t%s\n' % (song, annot))

    return out_list


def make_file_list(gtzan_path, prng, n_folds=5, songs_per_genre=None):
    """
    Generates lists
    """
    audio_path = os.path.join(gtzan_path, 'audio')
    out_path = os.path.join(gtzan_path, 'lists')
    files_list = []
    for ext in ['.au', '.mp3', '.wav']:
        files = U.getFiles(audio_path, ext)
        files_list.extend(files)

    annotations = get_annotations(files_list)

    if songs_per_genre is not None:
        # select only x songs per genre
        # create a dictionary {genre1: [song1, song2], genre2: [song3, song4]}
        genres_dic = {}
        for k, v in annotations.iteritems():
            genres_dic[v] = genres_dic.get(v, [])
            genres_dic[v].append(k)
        files_list = []
        for k in genres_dic.iterkeys():
            sample = prng.choice(genres_dic[k], size=songs_per_genre, replace=False)
            print "Selected %i songs for %s" % (len(sample), k)
            files_list.extend(sample)

    prng.shuffle(files_list)  # shuffle at the end of the selection
    annotations = get_annotations(files_list)  # update annotations

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    audio_list_path = os.path.join(out_path, 'audio_files.txt')
    open(audio_list_path, 'w').writelines(['%s\n' % f for f in files_list])

    ground_truth_path = os.path.join(out_path, 'ground_truth.txt')
    open(ground_truth_path, 'w').writelines(
        generate_mirex_list(files_list, annotations))
    generate_ground_truth_pickle(ground_truth_path)

    folds = get_folds(files_list, n_folds=n_folds)

    # Single fold for quick experiments
    create_fold(0, 1, folds, annotations, out_path)

    for n in range(n_folds):
        create_fold(n, n_folds, folds, annotations, out_path)


def create_fold(n, n_folds, folds, annotations, out_path):
    train_path = os.path.join(
        out_path, 'train_%i_of_%i.txt' % (n + 1, n_folds))
    valid_path = os.path.join(
        out_path, 'valid_%i_of_%i.txt' % (n + 1, n_folds))
    test_path = os.path.join(out_path, 'test_%i_of_%i.txt' % (n + 1, n_folds))

    test_list = folds[n]
    train_list = []
    for m in range(len(folds)):
        if m != n:
            train_list.extend(folds[m])

    open(train_path, 'w').writelines(
        generate_mirex_list(train_list, annotations))
    open(test_path, 'w').writelines(
        generate_mirex_list(test_list, annotations))
    split_list_file(train_path, train_path, valid_path, ratio=0.8)


def split_list_file(input_file, out_file1, out_file2, ratio=0.8):
    input_list = open(input_file, 'r').readlines()

    n = len(input_list)
    nsplit = int(n * ratio)

    list1 = input_list[:nsplit]
    list2 = input_list[nsplit:]

    open(out_file1, 'w').writelines(list1)
    open(out_file2, 'w').writelines(list2)


def get_annotation(filename):
    genre = os.path.split(U.parseFile(filename)[0])[-1]
    return genre


def get_annotations(files_list):
    annotations = {}
    for filename in files_list:
        annotations[filename] = get_annotation(filename)

    return annotations


def generate_ground_truth_pickle(gt_file):
    gt_path, _ = os.path.split(gt_file)
    tag_file = os.path.join(gt_path, 'tags.txt')
    gt_pickle = os.path.join(gt_path, 'ground_truth.pickle')

    lines = open(gt_file, 'r').readlines()

    tag_set = set()
    for line in lines:
        filename, tag = line.strip().split('\t')
        tag_set.add(tag)
    tag_list = sorted(list(tag_set))
    open(tag_file, 'w').writelines('\n'.join(tag_list + ['']))

    tag_dict = dict([(tag, i) for i, tag in enumerate(tag_list)])
    n_tags = len(tag_dict)

    mp3_dict = {}
    for line in lines:
        filename, tag = line.strip().split('\t')
        tag_vector = mp3_dict.get(filename, numpy.zeros(n_tags))
        if tag != '':
            tag_vector[tag_dict[tag]] = 1.
        mp3_dict[filename] = tag_vector
    pickle.dump(mp3_dict, open(gt_pickle, 'w'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Creates the lists for training/validation/test data.")
    parser.add_argument("dataset_dir", help="/path/to/dataset_dir")
    parser.add_argument("-f", "--folds", type=int, default=10, help="number of folds")
    parser.add_argument("-s", "--songs_per_genre", type=int, default=None, help="number of songs per genre to use")
    parser.add_argument("-r", "--seed", type=int, default=None, help="set a specific seed")
    args = parser.parse_args()

    prng = RandomState(args.seed)

    print "Seed: %i" % prng.get_state()[1][0]  # ugly but works in numpy 1.8.1

    make_file_list(
        os.path.abspath(args.dataset_dir),
        prng,
        args.folds,
        args.songs_per_genre)
