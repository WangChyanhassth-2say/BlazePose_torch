# BlazePose-Full

To the full version I tried to implement totally follwed by the paper.  
But to the output I use [SimDR](https://github.com/leeyegy/SimDR).  
You can run the lib/models/BlazePose.py to check the model layers.  
The params goes to 3.4m(for 17 keypoints), which is quite similar to the one in the paper(3.5m for 32 kerpoints).  


The training code you may refer [HRNet](https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation), you need to replace the folders by those I uploded here.  
If you want to train the full version model, you may need to delete the regression layers and train the heatmap branch as pretrain.  
Or you can use the tools/model_first.pth which I given [here](https://github.com/WangChyanhassth-2say/BlazePose_torch/blob/main/full/tools/model_first.pth).

Training for 210 epoch:  
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| BlazePose-Full | 0.653 | 0.883 | 0.725 | 0.625 | 0.697 | 0.690 | 0.898 | 0.754 | 0.654 | 0.743 | 
