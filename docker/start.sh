#!/bin/bash
cat /welcome.txt
export PYTHONUNBUFFERED=1

if [[ ! -f "v2-inference-v.yaml" ]]; then
    python utils/get_yamls.py
fi

mkdir -p logs input

# RunPod SSH
if [[ -v "PUBLIC_KEY" ]] && [[ ! -d "${HOME}/.ssh" ]]
then
    pushd $HOME
    mkdir -p .ssh
    echo ${PUBLIC_KEY} > .ssh/authorized_keys
    chmod -R 700 .ssh
    popd
    service ssh start
fi

tensorboard --logdir /workspace/EveryDream2trainer/logs --host 0.0.0.0 &

# RunPod JupyterLab
if [[ $JUPYTER_PASSWORD ]]
then
    jupyter nbextension enable --py widgetsnbextension
    jupyter labextension disable "@jupyterlab/apputils-extension:announcements"
    jupyter lab --allow-root --no-browser --port=8888 --ip=* --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=/workspace/EveryDream2trainer
else
    echo "Container Started"
    sleep infinity
fi
