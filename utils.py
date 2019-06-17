import os
import shutil
import numpy as np
import progressbar as pbar
import sys
import time
import logging

def split_data(root, copy_dir, val_rate = 0.3, test_rate = 0.1):
    classes = os.listdir(root)
    floders = ['train', 'val', 'test']
    for x in floders:
        x_floder = os.path.join(copy_dir, x)
        if os.path.exists(x_floder):
            shutil.rmtree(x_floder)
        os.mkdir(x_floder)

        for class_name in classes:
            os.mkdir(os.path.join(x_floder, class_name))

    widgets = ['Progress: ', pbar.Percentage(), ' ', pbar.Bar('#'), ' ', pbar.Timer(),
               ' ', pbar.ETA(), ' ', pbar.FileTransferSpeed()]
    total = len(classes)
    bar = pbar.ProgressBar().start()

    for i, class_name in enumerate(classes):
        bar.update(i)
        files = os.listdir(os.path.join(root, class_name))
        np.random.shuffle(files)

        val_num = int(len(files)*val_rate)
        test_num = int(len(files)*test_rate)

        val_files = files[0:val_num]
        test_files = files[val_num:val_num+test_num]
        train_files = files[val_num+test_num:]

        cp_files = {'train': train_files, 'test': test_files, 'val': val_files}
        src_folder = os.path.join(root, class_name)

        for x in floders:
            dst_folder = os.path.join(copy_dir, x)
            dst_folder = os.path.join(dst_folder, class_name)
            for f in cp_files[x]:
                shutil.copy(os.path.join(src_folder, f),
                            os.path.join(dst_folder, f))

TOTAL_BAR_LENGTH = 80
LAST_T = time.time()
BEGIN_T = LAST_T

def progress_bar(current, total, msg=None):

    global LAST_T, BEGIN_T
    if current == 0:
        BEGIN_T = time.time()  # Reset for new bar.

    current_len = int(TOTAL_BAR_LENGTH * (current + 1) / total)
    rest_len = int(TOTAL_BAR_LENGTH - current_len) - 1

    sys.stdout.write(' %d/%d' % (current + 1, total))
    sys.stdout.write(' [')
    for i in range(current_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    current_time = time.time()
    step_time = current_time - LAST_T
    LAST_T = current_time
    total_time = current_time - BEGIN_T

    time_used = '  Step: %s' % format_time(step_time)
    time_used += ' | Tot: %s' % format_time(total_time)
    if msg:
        time_used += ' | ' + msg

    msg = time_used
    sys.stdout.write(msg)

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def creat_logger(log_file,tty = True):
    # 创建log
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)

    # 设置格式
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s]: %(message)s')

    # 创建指向log_file的句柄
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    # 添加句柄
    logger.addHandler(file_handler)

    if tty == True:
        # 创建指向屏幕显示的句柄
        stream = logging.StreamHandler()
        stream.setLevel(logging.DEBUG)
        stream.setFormatter(formatter)
        # 添加句柄
        logger.addHandler(stream)

    return logger

if __name__ == "__main__":
    split_data('../256_ObjectCategories','../test',0.2,0.1)
