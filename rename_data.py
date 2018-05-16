import os, pandas as pd
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('root_dir', '.', "root dir of data")
flags.DEFINE_integer('items_to_use', 3, "number of last items of filepaths to retain")
flags.DEFINE_string('saving_name', 'data.csv', "saving name for csv file covering all the data")
flags.DEFINE_string('data_file_name', 'driving_log.csv', "name of data log in subdirs")
flags.DEFINE_string('col_names', 'center-left-right-steering-throttle-brake-speed', "column names")
flags.DEFINE_string('path_sep', '\\', "path_separator in initial logs")

root_dir = FLAGS.root_dir
items_to_use = FLAGS.items_to_use
saving_name = FLAGS.saving_name
data_file_name = FLAGS.data_file_name
col_names = FLAGS.col_names.split('-')
path_sep = FLAGS.path_sep

def change_path(df, true_path, last_items_to_use, path_sep="\\"):
    result = df.copy()
    for camera in ['center', 'left', 'right']:
        true_paths = []
        paths = result[camera].values
        for path in paths:
            a = os.path.join(true_path, *path.split(path_sep)[-last_items_to_use:])
            true_paths.append(a)
        result[camera] = true_paths
    return result

def make_csv_from_several(root_dir, items_to_use, saving_name,
                          data_file_name = 'driving_log.csv',
                          col_names = ['center','left','right','steering','throttle','brake','speed'],
                         path_sep='\\'):
    folders = os.listdir(root_dir)
    data_files = []
    for folder in folders:
        data_files.append(os.path.join(root_dir, folder, data_file_name))
    df = None
    for file in data_files:
        data = pd.read_csv(file, names = col_names)
        ch_df = change_path(data, root_dir, items_to_use, path_sep=path_sep)
        if df is None:
            df = ch_df
        else:
            df = pd.concat([df, ch_df], ignore_index=True)
    df.to_csv(saving_name, index=False)
    
make_csv_from_several(root_dir, items_to_use, saving_name,
                      data_file_name = data_file_name,
                      col_names = col_names,
                      path_sep = path_sep)
