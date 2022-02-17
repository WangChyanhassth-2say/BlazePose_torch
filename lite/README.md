# BlazePose-Lite

To the lite version I tried to implement all by heatmap branch.  
You can run the BlazePose.py to check the model layers.  
The params goes to 1.7m, which is a little bit larger than the model proposed in the paper(1.3m).  


The training code you may refer [HRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation), you only need to add the model and change some configs.

Training for 210 epoch and using [DARK](https://github.com/ilovepose/DarkPose) to decode the heatmap:  
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| BlazePose-Lite | 0.623 | 0.894 | 0.714 | 0.616 | 0.684 | 0.679 | 0.906 | 0.748 | 0.648 | 0.726 |  

