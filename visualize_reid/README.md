# Background Knowledge

Cloned from pyvision.

[Here](https://drive.google.com/drive/folders/1zOnaYQDOdhqWjW9ccykS4i8NwtDBUlTk?usp=sharing) is some useful material that you can use in order to understand the mathematics behind this project.
[Here](https://www.youtube.com/watch?v=RDkwklFGMfo&list=PLTBdjV_4f-EJn6udZ34tht9EVIW7lbeo4&ab_channel=cvprtum) are youtube videos from the Multiple View Geometry Course - Lectures 2 to 5 are explaining everything that is needed for this project.  

# Key concepts

**_Perspective projection_** - how to project a world point (3D) onto the image plane (2D)

**_Intrinsics of a camera_** - parameters that are camera dependent. These include the focal length, the principle point, the skew and the scale offset. 

**_Extrinsics of a camera_** - how the camera moves in reference to the world. Keep in mind that the world as well as the camera have their own coordinate systems. So, we need a way to describe the relationship between these two coordinate frames. Since the camera is a rigid body - it can only rotate and translate, rotation and translation of the camera are used as the extrinsic parameters that describe how the camera is positioned w.r.t. the world.

**_World point_** - a 3D point from the scene that we are capturing 

**_Point in camera coordinates_** - a 3D world point in the coordinate system of the camera

**_Pixel_** - a 2D point in the image 

# Key equation to understand
_```λx = K * (RX + T)```_, where:

_```X```_ = _```(X1, Y1, Z1)```_, a world point 

_```R```_ = Rotation matrix

_```T```_ = Translation vector

_```K```_ = Instrinsic matrix

_```x```_ = _```(x1, y1)```_, pixel coordinates in the image

_```λ```_ = depth 

Rotation and translation are applied to convert a world point to the camera coordinate system. Afterwards, the intrinsics (K) are applied to project the point in camera coodinates to the image plane. We are still in camera coordinates, not in pixels. In order to get to the pixel coordinates, the perspective projection is applied.

# visualize.py

## render_camera_poses()
![render_poses](assets/render_camera_poses.png)
This is the result after running the render_camera_poses function. The yellow sphere represents a 3D world point. The world consists of the point clouds gathered from all cameras. Also, the coordinate systems for all the cameras are displayed. The blue arrow represents how the respective camera "sees" the scene.

## render_single_pose()
![render_single_pose](assets/render_single_pose.png)
After running this function, we can see the pointcloud and the yellow sphere in two coordinate systems: the depth camera 02 coordinate system and the world coordinate system.

# visualize_3D_reprojections.py
![vis_3d_reprojection](assets/visualize_3d_reprojections.png)
This script provides a way to visualize 3D reprojections of corresponding pixels that represent "the center" of a person from one frame.  

