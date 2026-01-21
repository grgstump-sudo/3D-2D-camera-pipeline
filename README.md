# 3D-2D-camera-pipeline
Built in C# for Photoneo 3D cameras. Its purpose is to connect to the API and download aa simultainus stream of 2D color camera and 3D scanner. 2D is .png and .ply for 3D scan data.
Requires a propritary file to Photoneo scanners "WrapperCSharp.cs" along with "PhoXi_API.dll" both come with the camera and its software.

Then there are two custom python scripts to render .ply files into a 2d .png with aplha transparency.

Python Dependancys:
open3d (as of posting this open3d only works with Python V11)
tqdm
pillow
numpy
