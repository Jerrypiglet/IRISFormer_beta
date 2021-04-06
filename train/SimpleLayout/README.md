# SimpleLayout
Simple layout (room as a cuboid + camera) composer and visualizer in both 3D and screen space, with projection and differentiable depth rendering.

## Demo
Scene             |  Projection and masks
:-------------------------:|:-------------------------:
![](https://i.imgur.com/32nOpyE.png)  |  ![](https://i.imgur.com/rq24dhL.png)

Depth map             |  Projection 
:-------------------------:|:-------------------------:

![](https://i.imgur.com/MN03tdK.png) | 


## Installation
### matplotlib with widget support
```
conda install -y nodejs
pip install --upgrade jupyterlab
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter labextension install jupyter-matplotlib
jupyter nbextension enable --py widgetsnbextension\
```

### dependencies