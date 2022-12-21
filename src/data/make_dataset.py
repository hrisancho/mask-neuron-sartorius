import os

def make_dataset():
    try:
        os.system(f"mkdir ./data/sartorius-cell-instance-segmentation/")
        os.system(f"pip install -q kaggle")
        os.system(f"mkdir ~/.kaggle")
        os.system(f"cp ./kaggle.json ~/.kaggle/")
        os.system(f"chmod 600 ~/.kaggle/kaggle.json")
        os.system(f"kaggle competitions download -c sartorius-cell-instance-segmentation")
        os.system(f"unzip -qq sartorius-cell-instance-segmentation.zip -d ./data/sartorius-cell-instance-segmentation/")
    except:
        os.system(f"echo Произошла ошибка")
        os.system(f"echo Пожалуйста установите dataset вручную.")
        os.system(f"https://www.kaggle.com/competitions/sartorius-cell-instance-segmentation/data")

if __name__ == '__main__':
    make_dataset()
