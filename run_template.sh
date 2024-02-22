#! /bin/bash


if { conda env list | grep 'interpolation'; } >/dev/null 2>&1; then
    echo "env exists"
else
    conda env create -f env.yml
fi

# enter your code here


git add mlruns
git pull
git commit -m "update mlruns"
git push

shutdown


