# A Neural Algorithm of Artistic Style

- [https://arxiv.org/abs/1508.06576](https://arxiv.org/abs/1508.06576)
- [Breaking Down Leon Gatys’ Neural Style Transfer in PyTorch](https://towardsdatascience.com/breaking-down-leon-gatys-neural-style-transfer-in-pytorch-faf9f0eb79db)


## All results you can find here:
 
- [Основной рабочий jupyter notebook - NST_Gatys_working_notebook.ipynb](./NST_Gatys_working_notebook.ipynb)
- [Готовый к использованию jupyter notebook - Colab_working_notebook.ipynb](./Colab_working_notebook.ipynb)


## Results

#### First experiments with style image(2000 steps optimizing input image)
For the content image, I chose my own photo. It's me at the climbing wall in Moscow, Limestone.

Content Image

![](./example/content.jpg)

Style Image

![](./example/style.jpg)

###### Results(input image - content image).

Style weight = 1e4, 1e5, 1e6, 1e7.
Content weight(constant) = 1.
![](./results/style/collage_style.png)

###### Results(input image - Gaussian noise). 

Style weight = 1e4, 1e5, 1e6, 1e7.
Content weight(constant) = 1.
![](./results/style/collage_noise.png)


#### Other styles(1000 steps optimizing input image)

##### Style image №1

![](./example/style_1.jpg)

###### Results(input image - content image).

Style weight = 1e6, 1e7.
Content weight(constant) = 1.
![](./results/style_1/collage_style_1.png)


##### Style image №2

![](./example/style_2.jpg)

###### Results(input image - content image).

Style weight = 1e6, 1e7.
Content weight(constant) = 1.
![](./results/style_2/collage_style_2.png)


##### Style image №3

![](./example/style_3_sliced.jpg)

###### Results(input image - content image).

Style weight = 1e6, 1e7.
Content weight(constant) = 1.
![](./results/style_3/collage_style_3.png)


##### Style image №4

![](./example/style_4_sliced.jpg)

###### Results(input image - content image).

Style weight = 1e6, 1e7.
Content weight(constant) = 1.
![](./results/style_4/collage_style_4.png)


##### Style image №5

![](./example/style_5_sliced.jpg)

###### Results(input image - content image).

Style weight = 1e6, 1e7.
Content weight(constant) = 1.
![](./results/style_5/collage_style_5.png)