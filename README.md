# Rectangling_Panoramic_Images
my implementation of Rectangling Panoramic Images via Warping

基本完成。只是我搞不懂原论文中的1百万像素使用远古i7在1.5s内跑完是怎么做到的，我猜本算法的最小二乘法的矩阵有某种神秘的性质，可以更简单地计算出来。也可以尝试往并行QR分解去进行优化（可惜我不会），不过原文说的是在i7上的单核跑，想必作者一定有神必算法去解决这个最小二乘问题。

个人博客的论文精读和实现指南：[https://kegalas.top/p/rectangling-panoramic-images-via-warping%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB%E4%B8%8E%E5%A4%8D%E7%8E%B0/](https://kegalas.top/p/rectangling-panoramic-images-via-warping%E8%AE%BA%E6%96%87%E7%B2%BE%E8%AF%BB%E4%B8%8E%E5%A4%8D%E7%8E%B0/)

原论文：[https://people.csail.mit.edu/kaiming/publications/sig13pano.pdf](https://people.csail.mit.edu/kaiming/publications/sig13pano.pdf)

requirements:
- opencv-3.4.16
- cmake-3.27.6
- g++-13.1.0
- eigen-3.3.9
- opengl-3.3 (已经在include文件夹里，但是需要自己设置glfw3.dll等文件)
