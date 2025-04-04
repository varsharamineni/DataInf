# create the virtual env loading cineca-ai module
module load profile/deeplrn
moudle av cineca-ai
module load cineca-ai/4.3.0

python -m venv DataInf-env

# activate the created virtual env to install your python packages.
source DataInf-env/bin/activate
pip install -r requirements.txt
deactivate