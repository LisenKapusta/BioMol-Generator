Сначала выполнить команду:
 pip install -e .   

Для проверки химического взаимодействия белка и лиганда необходимо использовать модель DeepDTA
для этого выполнить :

Создать новое окружение
conda create -n deepdta python=3.6
conda activate deepdta

Установить библиотеки
pip install numpy==1.19.5
pip install pandas
pip install scikit-learn
pip install tensorflow==1.14
pip install keras==2.3.1
pip install h5py==2.10.0
pip install rdkit

склонировать репозиторий
git clone https://github.com/hkmshb/DeepDTA.git 