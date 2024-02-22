#! /bin/bash

if [[ $(conda --version) == "conda: command not found" ]]; then
    /opt/anaconda/condabin/conda init
    source ~/.bashrc

    source activate base
    conda install -n base conda-libmamba-solver -y
    conda config --set solver libmamba
fi

if { conda env list | grep 'interpolation'; } >/dev/null 2>&1; then
    echo "env exists"
else
    conda env create -f env.yml -y
fi
source activate interpolation

# enter your code here


git add mlruns
git pull
git commit -m "update mlruns"
git push

shutdown


