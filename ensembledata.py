import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)

    from algos import *
    import pandas as pd
    import numpy as np
    import os, time
    from data import Data
    from tqdm import tqdm
    from functools import partial
    import threading
    from multiprocessing import Process, Manager

def thread(start, end, df, index, thread_num, batch_size, total, n_threads):
    columns = ['name', 'bounds']
    data = Data()
    process_df = [0] * 12
    for i in range(len(process_df)):
        process_df[i] = pd.DataFrame(np.nan, index = range((end - start) // 4), columns = columns)
    with tqdm(total = end - start, desc = "Ensembling {0}".format(thread_num)) as pbar:
        for j in range((end - start) // batch_size + 1):
            data.get_data('train', start + j * batch_size, start + (j + 1) * batch_size if start + (j + 1) * batch_size < end else end)
            for i, img in enumerate(data.x):
                stats = [brightness(img), fourier_sharpness(img), canny_sharpness(img.astype(np.uint8))]
                if -1 in stats:
                    print("Failed at index {0} in batch {1}".format(i, j))
                    errors.append(i, j)
                    continue
                k = None
                if stats[0] < 0.203:
                    if stats[1] < 0.172:
                        if stats[2] < 0.0434:
                            k = 0
                        else:
                            k = 1
                    else:
                        if stats[2] < 0.0434:
                            k = 2
                        else:
                            k = 3
                if stats[0] > 0.203 and stats[0] < 0.382:
                    if stats[1] < 0.172:
                        if stats[2] < 0.0434:
                            k = 4
                        else:
                            k = 5
                    else:
                        if stats[2] < 0.0434:
                            k = 6
                        else:
                            k = 7
                if stats[0] > 0.382:
                    if stats[1] < 0.172:
                        if stats[2] < 0.0434:
                            k = 8
                        else:
                            k = 9
                    else:
                        if stats[2] < 0.0434:
                            k = 10
                        else:
                            k = 11
                
                process_df[k].loc[index[k]] = [data.names[i], data.bounds[i]]
                index[k] += 1
                pbar.update(1)
    for i, d in enumerate(process_df):
        df[i] = df[i].append(d.dropna(), ignore_index = True)

def ensemble(batch_size, total = -1, n_threads = 1):
    if total == -1:
        total = 70000
    path = "D/Documents/Sci-Inq"
    columns = ['name', 'bounds']
    classes = [0] * 12
    errors = []
    for i in range(len(classes)):
        classes[i] = pd.DataFrame(columns = columns)
    
    manager = Manager()
    df = manager.list(classes)
    index = manager.list([0] * 12)
    threads = []
    for i in range(n_threads):
        start = int(i / n_threads * total)
        end = int((i + 1) / n_threads * total)
        threads.append(Process(target = partial(thread, batch_size = batch_size, total = total, n_threads = n_threads), args = (start, end, df, index, i)))

    for i in threads:
        i.start()

    for i in threads:
        i.join()

    for i, c in enumerate(df):
        c.to_csv('image_classes/class_new' + str(i) + '.csv')

if __name__ == '__main__':
    ensemble(400, 70000, n_threads = 2)