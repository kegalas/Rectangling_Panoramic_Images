# Rectangling_Panoramic_Images
my implementation of Rectangling Panoramic Images via Warping

尚未完成，处于正在开发的过程中。

目前还差Line Preservation部分的能量没做。另外需要手动把img.vs和img.fs复制到可执行文件目录下，才能使用。后续可以研究下怎么用cmake自动实现这个。

后续会在个人博客更新论文精读和实现指南。

原论文：[https://people.csail.mit.edu/kaiming/publications/sig13pano.pdf](https://people.csail.mit.edu/kaiming/publications/sig13pano.pdf)

requirements:
- opencv-3.4.16
- cmake-3.27.6
- g++-13.1.0
- eigen-3.3.9
- opengl-3.3 (已经在include文件夹里，但是需要自己设置glfw3.dll等文件)
