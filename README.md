# CUDA exercise to generate disparity map

## Environments
* CUDA: 11.3
* Python: 3.7
* PyTorch: 1.10.0

## Code structure
```shell
├── include
│   └── sub3.h # header file of sub3 cuda kernel
├── kernel
│   └── sub3_kernel.cu # sub3 cuda kernel
├── pytorch
│   ├── sub3_ops.cpp # torch wrapper of sub3 cuda kernel
│   ├── setup.py # compile the kernal code
│   └── load_image.py # generate disparity map
├── LICENSE
└── README.md
```

## PyTorch
### Compile cpp and cuda
**Setuptools**  
```shell
python3 pytorch/setup.py install
```

### Run python
**Generate disparity image**  
```shell
python3 pytorch/load_image.py --min_disparity 1 --left_image <your_own_data or default> --right_image <your_own_data or default>
```