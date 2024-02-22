#! /bin/bash

if [[ $(conda config --show | grep "solver: classic") == "solver: classic" ]]; then
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


