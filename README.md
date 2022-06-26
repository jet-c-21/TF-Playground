# TF-Playground

## Environment
#### conda
```
conda create --name tf_plgd python=3.8 -y
```
```
pip install ipykernel
python -m ipykernel install --user --name tf_plgd --display-name "TF-Playground"
```

#### start jupyter lab
```shell
jupyter lab --ip 0.0.0.0 --port 8888 --allow-root
```