#Readme
## Modified VOneNet very Robust
This project is build based on CS2470 research, the foolproof eyes. In extend to it, we build another diffusion model and incorperate them together to genereate a better robustness result.\

E stands for EfficientNet.

V stands for VOneNet.

D stands for Diffusion Model.

![](setting.png)


![](main_result.png)

Diffustion and EfficientNet accuracy under different t:

![](diffusion_result.png)
For our experimental setting, we picked t=20 for its clear performance on the original image as well as a minimum trade-off between robustness and accuracy.

VOneNet under different number of Gabor Filters (GF):

![](vone_gf.jpeg)
For our experimental setting, we picked GF number to be 8 for compromised training accuracy and runtime.


