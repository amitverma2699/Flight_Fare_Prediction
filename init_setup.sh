echo [$(date)]: "START"


echo [$(date)]: "creating env with python 3.12 version" 


conda create --prefix ./myenv python=3.12.4 -y


echo [$(date)]: "activating the environment" 

source activate ./myenv

echo [$(date)]: "installing the dev requirements" 

pip install -r requirements.txt

echo [$(date)]: "END" 