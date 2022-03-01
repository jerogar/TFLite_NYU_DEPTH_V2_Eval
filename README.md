# TFLite_NYU_DEPTH_V2_Eval

Examples of Monocular Depth Estimation(MDE) using TensorFlow Lite C++ interpreter.

The official website is:

- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [TensorFlow Lite CMake build guide](https://www.tensorflow.org/lite/guide/build_cmake_arm)

## Build and Test environment

- Uduntu 18.04
- OpenCV 3.4.6, CMake 3.1.6

## Example

- Download NYU Depth v2 Dataset (from DenseDepth repo.:  https://github.com/ialhashim/DenseDepth)

    ```bash
     cd ${DB_path}
     wget https://s3-eu-west-1.amazonaws.com/densedepth/nyu_test.zip
     unzip nyu_test.zip 
    ```
    
    - convert *.npy to image and csv gt files using "res/convert_nyu_test.py" 
    
- Build

    ```bash
    cd ${source_path}
    vi CMakeLists.txt
    ```
    
    - Set OpenCV and TFLite Path
    
    ```bash
     set(TFLITE_BUILD_PATH ${TFLite path})
     set(OPENCV_INSTALL_PATH "${OpenCV install path}")
     # e.g. 
     set(TFLITE_BUILD_PATH "/media/Data/lib/tensorflow_src")
     set(OPENCV_INSTALL_PATH "/usr/local/")
    ```


    ```bash
     cd ${source_path}
     mkdir build
     cd build
  
     cmake ..
     make
    ```

- Run

    ```bash
     ./DepthEst_TfLite -ModelPath ${Model_Path} -dbPath #{DB_path} -savePath ${save_path}
     # e.g.
     ./DepthEst_TfLite -ModelPath ../res/model_float32.tflite -dbPath ../res/DenseDepth -savePath ../res/result
    ```