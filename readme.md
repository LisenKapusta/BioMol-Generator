Сначала выполнить команду:
 pip install -e .   

Для проверки химического взаимодействия белка и лиганда необходимо использовать модель DeepDTA
для этого выполнить :

Создать новое окружение
conda create -n deepdta python=3.6
conda activate deepdta

Установить библиотеки
pip install numpy==1.19.5 pandas scikit-learn tensorflow==1.14 h5py==2.10.0 python-docx keras rdkit-pypi tensorflow matplotlib

склонировать репозиторий
git clone https://github.com/hkmshb/DeepDTA.git 