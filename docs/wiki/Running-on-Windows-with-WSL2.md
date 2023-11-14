---
layout: default
title: Running on Windows with WSL2
parent: Wiki
has_children: false
nav_order: 16
---

The framework does not run natively on Windows, but it runs with WSL2 on Windows. Hence, you need to clone the repository within WSL2. An X-Server is required. It is important that you use OpenGL >= 1.5, which you can achieve if you follow the instructions [here](https://linuxtut.com/en/2841f1f15d364c2377a1/).
Furthermore, it is important that you don't use the Windows file system but the WSL internal one. This is because of some symlinks which don't work otherwise. This means, that you cannot clone the system anywhere under `/mnt/<Windows drive Letter>`, but you should clone it into `<home>/<USERNAME>/` instead.

If you want to display a video window for rendering, you need to export the DISPLAY variable appropriately. To find the appropriate IP address of your X-Server, you can open a PowerShell, type `ipconfig` and look up the IP address of the WSL virtual ethernet adapter (not your default hardware network adapter). 

You can perform the export by running `export DISPLAY=<ip-address>:0` in the linux console.

Another issues is the file format. If you just clone the repository, all files might have DOS-basesd line endings. To fix this, use `dos2unix`. For example, to fix all files recursively, you ould run the following command in your Scilab-RL directory: 
`find . -type f -print0 | xargs -0 -n 50 -P $(nproc) dos2unix`