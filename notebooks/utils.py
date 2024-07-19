import os
import librosa as lb
import random
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from pydub import AudioSegment
type_list = {0: ["cailuong", "CaiLuong"], 1: ["catru", "Catru"], 2:["chauvan", "Chauvan"], 3: ["cheo", "Cheo"], 4: ["hatxam", "Xam"]}
def load_sample_directories(root, type_index, samples_list , num_of_samples, mode="random"):
    # Mode:
    # Random: Load random dir
    # All: Load all dir
    # Return:
    # Sample_list: Dictionary {index: {"dir": "/...."}}
    # Define a function for padding index
    def padding(index):
        return str(index).zfill(3)  # Zero-pad the index to ensure three digits

    if mode == "random":
        # Load random samples
        random_indices = np.random.randint(0, 500, size=num_of_samples)
        for index in random_indices:
            dir_index = padding(index)
            dir_path = os.path.join(root, type_list[type_index][0], f"{type_list[type_index][1]}.{dir_index}.wav")
            samples_list[index] = {"dir": dir_path}

    elif mode == "all":
        # Load all samples
        for i in range(num_of_samples):
            dir_index = padding(i)
            dir_path = os.path.join(root, type_list[type_index][0], f"{type_list[type_index][1]}.{dir_index}.wav")
            samples_list[i] = {"dir": dir_path}

    return samples_list


def load_samples(samples_listdir):
    """
    Load and sampling
    Input: samples_listdir - Dictionary {index: {"dir": "/...."}}
    Output: samples_listdir - Dictionary {index: {"dir": "/....", "sampling": array}}
    """
    for index, sample in samples_listdir.items():
        try:
            file, sr = lb.load(sample["dir"])
            if len(samples_listdir[index]) == 1:  # Avoid adding multiple times
                samples_listdir[index]["sampling"] = file
        except FileNotFoundError:
            print(f"File not found: {sample['dir']}. Skipping...")
    return samples_listdir


def get_stft(samples, n_fft=2048, hop_length=512):
    """
    Input: samples: {index: {"dir": "/..."}}
    Output: samples: {index: {"dir": "/...", "stft:" array}}
    """
    for index, item in samples.items():
        if 'sampling' in item:
            D = np.abs(lb.stft(item["sampling"], n_fft=n_fft, hop_length=hop_length))
            samples[index]["stft"] = D
    return samples

def plot_fft(samples, type_index):
    """
    Get frequency domain representation
    """
    for index, item in samples.items():
        plt.figure(figsize = (16, 6))
        plt.plot(item["stft"])
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title("STFT of sample {} of class {}".format(index, type_list[type_index][0]))
def plot_mfcc(samples, type_index):
    """
    Get frequency domain representation
    """
    for index, item in samples.items():
        plt.figure(figsize = (16, 6))
        plt.plot(item["mfcc"])
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        plt.title("mfcc of sample {} of class {}".format(index, type_list[type_index][0]))

def get_mel_spectrogram(samples, type_index, sr=22050):
    """
    Get mel-spectrogram (db)
    Input: {index: {"dir": "/...", "stft": array, }}
    Output: {index: {"dir": "/...", "stft": array, "mel-spec-db": array}}
    """
    for index, item in samples.items():
        if "sampling" in item:
            S = lb.feature.melspectrogram(y=item["sampling"], sr=sr)
            samples[index]["mel_spec"] = S
    return samples
def get_log_mel_spectrogram(samples, type_index, sr=22050):
    """
    Get log-mel-spectrogram (db)
    Input: {index: {"dir": "/...", "stft": array, }}
    Output: {index: {"dir": "/...", "stft": array, "mel-spec-db": array}}
    """
    for index, item in samples.items():
        if "sampling" in item:
            S = lb.feature.melspectrogram(y=item["sampling"], sr=sr)
            S_db = lb.amplitude_to_db(S, ref=np.max)
            samples[index]["mel-spec-db"] = S_db
    return samples
def plot_log_mel_spectrogram(samples, type_index, sr = 22050, HOP_LENGTH = 512):
    """
    Plot log-mel-spectrogram
    """
    for index, item in samples.items():
        S_DB = item["mel-spec-db"]
        plt.figure(figsize = (16, 6))
        lb.display.specshow(S_DB, sr=sr, hop_length = HOP_LENGTH, x_axis = 'time', y_axis = 'log')
        plt.colorbar()
        plt.title("Mel Spectrogram of sample {} of class {}".format(index, type_list[type_index][0]), fontsize = 20)

def plot_mel_spectrogram(samples, type_index, sr = 22050, HOP_LENGTH = 512):
    """
    Plot log-mel-spectrogram
    """
    for index, item in samples.items():
        S_DB = item["mel_spec"]
        plt.figure(figsize = (16, 6))
        lb.display.specshow(S_DB, sr=sr, hop_length = HOP_LENGTH, x_axis = 'time', y_axis = 'log')
        plt.colorbar()
        plt.title("Mel Spectrogram of sample {} of class {}".format(index, type_list[type_index][0]), fontsize = 20)

def save_log_mel_spec(samples, root, type_index):
    """
    save log-mel-spec
    After running, images of a class will be saved in: root/class/file_name.png
    """
    for index, item in samples.items():
        try:
            S_db = item["mel-spec-db"]
            folder_root_log_mel_images = os.path.join(root, type_list[type_index][0])  # Using os.path.join
            if not os.path.exists(folder_root_log_mel_images):
                os.makedirs(folder_root_log_mel_images)
            # Get file name from fir
            file_name = os.path.splitext(os.path.basename(item["dir"]))[0]  # Getting file name without extension
            file_path = os.path.join(folder_root_log_mel_images, "{}.png".format(file_name))  # Using os.path.join
            plt.imsave(file_path, S_db)
        except KeyError:
            print("Skipping item because required keys are missing.")

def save_stft(samples, root, type_index):
    """
    save log-mel-spec
    After running, images of a class will be saved in: root/class/file_name.png
    """
    for index, item in samples.items():
        try:
            s_stft = item["stft"]
            folder_root_stft = os.path.join(root, type_list[type_index][0])  # Using os.path.join
            if not os.path.exists(folder_root_stft):
                os.makedirs(folder_root_stft)
            # Get file name from fir
            file_name = os.path.splitext(os.path.basename(item["dir"]))[0]  # Getting file name without extension
            file_path = os.path.join(folder_root_stft, "{}.png".format(file_name))  # Using os.path.join
            plt.imsave(file_path, s_stft)
        except KeyError:
            print("Skipping item because required keys are missing.")

def save_mel_spec(samples, root, type_index):
    """
    save log-mel-spec
    After running, images of a class will be saved in: root/class/file_name.png
    """
    for index, item in samples.items():
        try:
            S = item["mel_spec"]
            folder_root_mel_images = os.path.join(root, type_list[type_index][0])  # Using os.path.join
            if not os.path.exists(folder_root_mel_images):
                os.makedirs(folder_root_mel_images)
            # Get file name from fir
            file_name = os.path.splitext(os.path.basename(item["dir"]))[0]  # Getting file name without extension
            file_path = os.path.join(folder_root_mel_images, "{}.png".format(file_name))  # Using os.path.join
            plt.imsave(file_path, S)
        except KeyError:
            print("Skipping item because required keys are missing.")

def get_mfcc(samples, sr=22050, n_mfcc=13):
    """
    Trích xuất MFCC từ dữ liệu âm thanh
    Input: samples: {index: {"dir": "/...", "sampling": array}}
           sr: Tần số lấy mẫu
           n_mfcc: Số lượng hệ số MFCC cần trích xuất
    Output: {index: {"dir": "/...", "mfcc": array}}
    """
    for index, item in samples.items():
        if "sampling" in item:
            y = item["sampling"]
            mfcc = lb.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
            samples[index]["mfcc"] = mfcc
    return samples

def save_mfcc(samples, root, type_index):
    """
    Lưu MFCC dưới dạng hình ảnh
    Sau khi chạy, các hình ảnh của mỗi class sẽ được lưu tại: root/class/file_name.png
    Input: samples: {index: {"dir": "/...", "mfcc": array}}
           root: Thư mục gốc
    """
    for index, item in samples.items():
        try:
            mfcc = item["mfcc"]
            folder_root_mfcc = os.path.join(root, type_list[type_index][0])  # Thư mục chứa hình ảnh MFCC
            if not os.path.exists(folder_root_mfcc):
                os.makedirs(folder_root_mfcc)
            # Lấy tên file từ đường dẫn
            file_name = os.path.splitext(os.path.basename(item["dir"]))[0]
            file_path = os.path.join(folder_root_mfcc, "{}.png".format(file_name))
            # Hiển thị và lưu hình ảnh MFCC
            plt.figure(figsize=(10, 4))
            plt.imshow(mfcc, aspect='auto', origin='lower')
            plt.axis('off')  # Tắt trục
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            
        except KeyError:
            print("Skipping sample")

def train_val_test_split(folder_root, dataset_root, type_index):
    """
    Split and save train/val/test set
    Input:
    
folder_root: folder_root containing mel-spec images
dataset_root: Directory to save dataset
type_root : train_root, val_root or test_root
type_index: class index in type_list
"""

    def save_set(subset, dataset_root, typeset, type_index):
      """
      Save X_train, X_val, X_test to their respective dir
      Input:
      """
      # Copy file from subset to train/val/test folder
      for file in subset:
          srcpath = os.path.join(src_dir, file)
          dst_dir = dataset_root + "/" + typeset + "/{}".format(type_list[type_index][0])
          if not os.path.exists(dst_dir):
              os.makedirs(dst_dir)
          shutil.copy(srcpath, dst_dir)

    src_dir = folder_root + "/{}".format(type_list[type_index][0])
    X = os.listdir(src_dir)
    Y = ["{}".format(type_list[type_index][0]) for i in range(0, len(X))]
    # Train 80%, test 20%
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.2, random_state=42, shuffle = True)
    # Val 10 %, test 10%
    X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size = 0.5, random_state=42, shuffle = True)

    # Create dataset_root to save dataset
    if not os.path.exists(dataset_root):
      os.makedirs(dataset_root)
    # Save train/val/test of each class
    save_set(X_train, dataset_root, "train", type_index)
    save_set(X_val, dataset_root, "val", type_index)
    save_set(X_test, dataset_root, "test", type_index)



