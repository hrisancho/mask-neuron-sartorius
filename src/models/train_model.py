import sys
sys.path.insert(0, '../')
from conf import DSAMPLE_SUBMISSION, TRAIN_CSV, TRAIN_PATH, TEST_PATH, TRAIN_SEMI_SUPERVISED_PATH, df_train

import os
import cv2
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from albumentations import *
from albumentations.pytorch import ToTensorV2

warnings.simplefilter('ignore')
tqdm.pandas()

# Модель: U-net
import torch
import collections.abc as container_abcs
torch._six.container_abcs = container_abcs
import segmentation_models_pytorch as smp

RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda:0" if (torch.cuda.is_available() ) else "cpu")
#(336, 336)
#(224, 224)
IMAGE_RESIZE = (256, 256)
LEARNING_RATE = 5e-4
EPOCHS = 500

# Задаём стартовое число для псевдо генератор случайных чисел. К примеру год создания этого ноутбука)
def fix_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

fix_all_seeds(2022)

# Набор обучающих данных и загрузчик данных
def rle_decode(mask_rle, shape, color=1):
    '''
    mask_rle: длина прогона в виде строки (начальная длина)
    shape: (высота, ширина, каналы) возвращаемого массива
    color: цвет для маски
    Returns Масив numpy (3-х мерная маска). Для статистики.
    '''
    s = mask_rle.split()

    starts = list(map(lambda x: int(x) - 1, s[0::2]))
    lengths = list(map(int, s[1::2]))
    ends = [x + y for x, y in zip(starts, lengths)]

    img = np.zeros((shape[0] * shape[1], shape[2]), dtype=np.float32)

    for start, end in zip(starts, ends):
        img[start : end] = color

    return img.reshape(shape)

def bin_decode(mask_rle, shape, color=1):
    '''
    mask_rle: длина прогона в виде строки (начальная длина)
    shape: (высота, ширина) возвращаемого массива
    return: пустой массив, 1 - маска, 0 - фон

    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color
    return img.reshape(shape)

def bin_build_masks(df_train, image_id, input_shape):
    height, width = input_shape
    labels = df_train[df_train["id"] == image_id]["annotation"].tolist()
    mask = np.zeros((height, width))
    for label in labels:
        mask += bin_decode(label, shape=(height, width))
    mask = mask.clip(0, 1)
    return mask

# Dataset и его загрущик данных
class CellDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.base_path = TRAIN_PATH
        self.transforms = Compose([Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1]),
                                   Normalize(mean=RESNET_MEAN, std=RESNET_STD, p=1),
                                   HorizontalFlip(p=0.5),
                                   VerticalFlip(p=0.5),
                                   ToTensorV2()])
        self.gb = self.df.groupby('id')
        self.image_ids = df.id.unique().tolist()

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        df = self.gb.get_group(image_id)
        annotations = df['annotation'].tolist()
        image_path = os.path.join(self.base_path, image_id + ".png")
        image = cv2.imread(image_path)
        mask = bin_build_masks(df_train, image_id, input_shape=(520, 704))
        mask = (mask >= 1).astype('float32')
        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        return image, mask.reshape((1, IMAGE_RESIZE[0], IMAGE_RESIZE[1]))

    def __len__(self):
        return len(self.image_ids)

ds_train = CellDataset(df_train)
image, mask = ds_train[1]
image.shape, mask.shape

dl_train = DataLoader(ds_train, batch_size=64, num_workers=4, pin_memory=True, shuffle=False)

# получаем группу из загрущика данных
batch = next(iter(dl_train))
images, masks = batch

# Функции потерь
def dice_loss(input, target):
    input = torch.sigmoid(input)
    smooth = 1.0
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    return ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Целевой размер ({}) должен совпадать с исходным размером ({})"
                             .format(target.size(), input.size()))
        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class MixedLoss(nn.Module):
    def __init__(self, alpha, gamma):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)

    def forward(self, input, target):
        loss = self.alpha*self.focal(input, target) - torch.log(dice_loss(input, target))
        return loss.mean()

# U-Net установка
model = smp.Unet("resnet34", encoder_weights="imagenet", activation=None).to(DEVICE)

# Тренировочный цикл
# На данный момент никаких проверок качества обучения или k-folds.
n_batches = len(dl_train)
model.train()

criterion = MixedLoss(10.0, 2.0)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(1, EPOCHS + 1):
    print(f"Текущая эпоха обучения: {epoch} / {EPOCHS}")
    running_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dl_train):

        # Предсказатель
        images, masks = batch
        images, masks = images.to(DEVICE),  masks.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Обратное распростонение ошибки
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    epoch_loss = running_loss / n_batches
    print(f"Эпоха: {epoch} - значение целевой фукции {epoch_loss:.4f}")

# Тестовый набор данных и загрущик данных
class TestCellDataset(Dataset):
    def __init__(self):
        self.test_path = TEST_PATH
        self.image_ids = [f[:-4]for f in os.listdir(self.test_path)]
        self.num_samples = len(self.image_ids)
        self.transform = Compose([Resize(IMAGE_RESIZE[0], IMAGE_RESIZE[1]), Normalize(mean=RESNET_MEAN, std=RESNET_STD, p=1), ToTensorV2()])

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.test_path, image_id + ".png")
        image = cv2.imread(path)
        image = self.transform(image=image)['image']
        return {'image': image, 'id': image_id}

    def __len__(self):
        return self.num_samples

ds_test = TestCellDataset()
dl_test = DataLoader(ds_test, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

# Постобработка: отдельные компоненты маски предсказания
def post_process(probability, threshold=0.5, min_size=300):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = []
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            a_prediction = np.zeros((520, 704), np.float32)
            a_prediction[p] = 1
            predictions.append(a_prediction)
    return predictions

# Кодирование длин
def rle_encoding(x):
    dots = np.where(x.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return ' '.join(map(str, run_lengths))

def check_is_run_length(mask_rle):
    if not mask_rle:
        return True
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    start_prev = starts[0]
    ok = True
    for start in starts[1:]:
        ok = ok and start > start_prev
        start_prev = start
        if not ok:
            return False
    return True

def create_empty_submission():
    fs = os.listdir(f"{TEST_PATH}")
    df = pd.DataFrame([(f[:-4], "") for f in fs], columns=['id', 'predicted'])
    df.to_csv("submission.csv", index=False)

# Визуализация прогнозов
model.eval()

submission = []
submission_2 = []
for i, batch in enumerate(tqdm(dl_test)):
    preds = torch.sigmoid(model(batch['image'].to(DEVICE)))
    preds = preds.detach().cpu().numpy()[:, 0, :, :]
    last_id_img = 0
    img_id = -1
    for image_id, probability_mask in zip(batch['id'], preds):
        try:
            if probability_mask.shape != IMAGE_RESIZE:
                probability_mask = cv2.resize(probability_mask, dsize=IMAGE_RESIZE, interpolation=cv2.INTER_LINEAR)
            probability_mask = cv2.resize(probability_mask, dsize=(704, 520), interpolation=cv2.INTER_LINEAR)
            predictions = post_process(probability_mask)
            for prediction in predictions:
                #plt.imshow(prediction)
                #plt.show()
                try:
                    submission.append((image_id, rle_encoding(prediction)))
                    if last_id_img == image_id:
                      submission_2[img_id][1] += f" {rle_encoding(prediction)}"
                    else:
                      submission_2.append([image_id, rle_encoding(prediction)])
                      last_id_img = image_id
                      img_id += 1
                except:
                    print("Error in RL encoding")
        except Exception as e:
            print(f"Исключение для img: {image_id}: {e}")

        # Заполнить изображения без прогнозов
        image_ids = [image_id for image_id, preds in submission]
        if image_id not in image_ids:
            submission.append((image_id, ""))

df_submission = pd.DataFrame(submission, columns=['id', 'predicted'])
df_submission.to_csv('submission.csv', index=False)
df_test = pd.DataFrame(submission_2, columns=['id', 'annotation'])
df_test.to_csv('submission_final.csv', index=False)

if df_submission['predicted'].apply(check_is_run_length).mean() != 1:
    print("Не удалось проверить длину цикла")
    create_empty_submission()

def plot_masks(image_id, colors:bool):
    labels = df_test[df_test["id"] == image_id]["annotation"].tolist()
    mask = np.zeros((520, 704,3))
    for label in labels:
        mask += rle_decode(label, shape=(520, 704,3))
    mask = mask.clip(0, 1)

    image = cv2.imread(f"{TEST_PATH}/{image_id}.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(16, 32))
    plt.subplot(3, 1, 1)
    plt.imshow(image)
    plt.axis("off")
    plt.subplot(3, 1, 2)
    plt.imshow(image)
    plt.imshow(mask, alpha=0.5)
    plt.axis("off")
    plt.subplot(3, 1, 3)
    plt.imshow(mask)
    plt.axis("off")

    plt.show();

plot_masks("7ae19de7bc2a", colors=False)

# Сохранение весов
torch.save(model.state_dict(), "./self-resnet34.pth")

