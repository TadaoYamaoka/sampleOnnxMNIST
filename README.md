# WindowsでのTensorRT MNISTサンプルプログラム

公式のサンプルプログラム[“Hello World” For TensorRT From ONNX](https://github.com/NVIDIA/TensorRT/tree/release/6.0/samples/opensource/sampleOnnxMNIST)を元にしたWindows上でTensorRTを動かすサンプルプログラムです。

モデルの訓練はPytorchで行い、ONNXでモデルを出力して、C++でモデルを読み込みTensorRTで推論を行います。

このサンプルプログラムではバッチサイズは固定長のみ対応しています。

PyTorchのMNISTの訓練は、PyTorch公式の[Basic MNIST Example](https://github.com/pytorch/examples/tree/master/mnist)を元にしています。
ONNXへの出力は、公式のチュートリアル[(optional) Exporting a Model from PyTorch to ONNX and Running it using ONNX](https://github.com/onnx/tutorials/blob/master/tutorials/PytorchOnnxExport.ipynb)を参考にしています。

## 環境
TensorRT 7.0.0と[互換性のあるバージョン](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/tensorrt-700/tensorrt-release-notes/tensorrt-7.html#rel_7-0-0)を使用する。

- TensorRT 7.0.0
- CUDA 10.2
- cuDNN 7.6.5
- PyTorch 1.3
- Visual Studio 2019
- Windows 10 64bit

PyTorch 1.3は、CUDA 10.2と互換性がないため、PyTorchではCUDA 10.1を使用する。
condaを使用してインストールを行うと、Pythonで使用されるCUDAを別にインストールできる。

例)
`conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.1 -c pytorch`

環境変数PATHにTensorRT 7.0.0の「lib」ディレクトリを追加する。

例)
`C:\TensorRT-7.0.0.11\lib`

## モデル訓練

```
cd train_mnist
python train_mnist
```

「..\data\MNIST」に「mnist.onnx」が保存される。

## ビルド
ビルド構成のプロットフォームを「x64」にする。

Visual Studio 2019で「sampleOnnxMNIST.sln」を開き、「sampleOnnxMNIST」プロジェクトのプロパティで、インクルードディレクトリとライブラリディレクトリをTensorRT 7.0.0とCUDA 10.2をインスタンスしたディレクトリに修正する。

ソリューションをビルドする。

## 推論実行
「sampleOnnxMNIST」プロジェクトを実行する。
成功すれば以下のように表示される。

```
&&&& RUNNING TensorRT.sample_onnx_mnist # H:\src\sampleOnnxMNIST\x64\Debug\sampleOnnxMNIST.exe
[03/12/2020-15:02:05] [I] Building and running a GPU inference engine for Onnx MNIST
----------------------------------------------------------------
Input filename:   ../data/mnist/mnist.onnx
ONNX IR version:  0.0.4
Opset version:    9
Producer name:    pytorch
Producer version: 1.3
Domain:
Model version:    0
Doc string:
----------------------------------------------------------------
[03/12/2020-15:02:05] [W] [TRT] Calling isShapeTensor before the entire network is constructed may result in an inaccurate result.
[03/12/2020-15:02:05] [W] [TRT] Calling isShapeTensor before the entire network is constructed may result in an inaccurate result.
[03/12/2020-15:02:07] [I] [TRT] Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
[03/12/2020-15:02:07] [I] [TRT] Detected 1 inputs and 1 output network tensors.
Input name : input
Output name : output
[03/12/2020-15:02:07] [W] [TRT] Current optimization profile is: 0. Please ensure there are no enqueued operations pending in this context prior to switching profiles
[03/12/2020-15:02:07] [I] Input:
[03/12/2020-15:02:07] [I]



               +@@@=
             -@@@@@%-
          .*@@@@@@@@*
         .@@@@#+:*@@+
         :%#%:  :%@@.
            .  :%@@+
              -*@@*
         :+%%%@@@@:
        *@@@@@@@@@@=
        #@@@@@@#+@@%
        *@@@#=    #@:
                  #@*
                  #@@.
                 .%@@.
      .###+      +@@#
      =@@@@#+:::-@@@+
      *@@@@@@@@@@@@#
      *@%#+#@@@@@@+.
      .@@@@@@@@@@%.
       .%@@@@@@+:





[03/12/2020-15:02:08] [W] [TRT] Explicit batch network detected and batch size specified, use execute without batch size instead.
[03/12/2020-15:02:08] [I] Output:
[03/12/2020-15:02:08] [I]  Prob 0  0.0000 Class 0:
[03/12/2020-15:02:08] [I]  Prob 1  0.0000 Class 1:
[03/12/2020-15:02:08] [I]  Prob 2  0.0000 Class 2:
[03/12/2020-15:02:08] [I]  Prob 3  1.0000 Class 3: **********
[03/12/2020-15:02:08] [I]  Prob 4  0.0000 Class 4:
[03/12/2020-15:02:08] [I]  Prob 5  0.0000 Class 5:
[03/12/2020-15:02:08] [I]  Prob 6  0.0000 Class 6:
[03/12/2020-15:02:08] [I]  Prob 7  0.0000 Class 7:
[03/12/2020-15:02:08] [I]  Prob 8  0.0000 Class 8:
[03/12/2020-15:02:08] [I]  Prob 9  0.0000 Class 9:
[03/12/2020-15:02:08] [I]
&&&& PASSED TensorRT.sample_onnx_mnist # H:\src\sampleOnnxMNIST\x64\Debug\sampleOnnxMNIST.exe
```