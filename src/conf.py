import os
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()

DSAMPLE_SUBMISSION  = "../../data/sartorius-cell-instance-segmentation/sample_submission.csv"
TRAIN_CSV = "../../data/sartorius-cell-instance-segmentation/train.csv"
TRAIN_PATH = "../../data/sartorius-cell-instance-segmentation/train"
TEST_PATH = "../../data/sartorius-cell-instance-segmentation/test"
TRAIN_SEMI_SUPERVISED_PATH = "../../data/sartorius-cell-instance-segmentation/train_semi_supervised"

df_train = pd.read_csv(TRAIN_CSV)

def getImagePaths(path):
    """
    Функция для объединения пути к каталогу(дериктории) с отдельными путями к изображениям(полным путём к конкретному изображению)

    parameters: path(string) - Путь к дериктории
    returns: image_names(string) - Список состоящий из полных путей к файлам
    """
    image_names = []
    for dirname, _, filenames in os.walk(path):
        for filename in tqdm(filenames):
            fullpath = os.path.join(dirname, filename)
            image_names.append(fullpath)
    return image_names


#Получите полные пути изображений для обучающих и тестовых наборов данных
train_images_path = getImagePaths(TRAIN_PATH)
test_images_path = getImagePaths(TEST_PATH)
train_semi_supervised_path = getImagePaths(TRAIN_SEMI_SUPERVISED_PATH)


