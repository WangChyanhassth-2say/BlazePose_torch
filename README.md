# BlazePose_torch  

torch implement of google blazePose  
BlazePose: On-device Real-time Body Pose tracking  


![image](https://github.com/WangChyanhassth-2say/BlazePose_torch/blob/main/lite/lite_out.gif)  
👆It's a 1920x1080 video runing by cpu only (Intel Core i5-1135G7) and showing the on-time fps.  


# Performance:
## BlazePose-Lite(1.7M Params)  
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| BlazePose | 0.623 | 0.883 | 0.714 | 0.616 | 0.684 | 0.679 | 0.898 | 0.748 | 0.648 | 0.726 |  

## BlazePose-Full(3.4M Params)  
| Arch | AP | Ap .5 | AP .75 | AP (M) | AP (L) | AR | AR .5 | AR .75 | AR (M) | AR (L) |
|---|---|---|---|---|---|---|---|---|---|---|
| BlazePose-Full | 0.653 | 0.894 | 0.725 | 0.625 | 0.697 | 0.690 | 0.906 | 0.754 | 0.654 | 0.743 | 
