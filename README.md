<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/infer_colorful_image_colorization/main/icon/icon.png" alt="Algorithm icon">
  <h1 align="center">infer_colorful_image_colorization</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/infer_colorful_image_colorization">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/infer_colorful_image_colorization">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/infer_colorful_image_colorization/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/infer_colorful_image_colorization.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

This algorithm enables the colorization of grayscale images using neural networks.


![input output colorization](https://richzhang.github.io/colorization/resources/images/teaser3.jpg)


## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow

```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_colorful_image_colorization", auto_connect=True)

# Run on your image  
wf.run_on(url="https://github.com/sczhou/CodeFormer/blob/master/inputs/gray_faces/Hepburn02.png?raw=true")

display(algo.get_input(0).get_image())
display(algo.get_output(0).get_image())
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **backend** (str) – default 'Default': Select run backend: "Default", Halide", "Inference engine", "OpenCV", "VKCOM", "CUDA".

- **target** (str) – default 'CPU': Select run target "CPU", "OpenCL FP32", "OpenCL FP16", "MYRIAD", "VULKAN", "FPGA", "CUDA FP32", "CUDA FP16".


```python
from ikomia.dataprocess.workflow import Workflow
from ikomia.utils.displayIO import display

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_colorful_image_colorization", auto_connect=True)

algo.set_parameters({
    "backend": "Default",
    "target": "CPU",
})

# Run on your image  
wf.run_on(url="https://github.com/sczhou/CodeFormer/blob/master/inputs/gray_faces/Hepburn02.png?raw=true")

# Inspect your results
display(algo.get_input(0).get_image())
display(algo.get_output(0).get_image())
```

## :mag: Explore algorithm outputs

Every algorithm produces specific outputs, yet they can be explored them the same way using the Ikomia API. For a more in-depth understanding of managing algorithm outputs, please refer to the [documentation](https://ikomia-dev.github.io/python-api-documentation/advanced_guide/IO_management.html).

```python
import ikomia
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
algo = wf.add_task(name="infer_colorful_image_colorization", auto_connect=True)

# Run on your image  
wf.run_on(url="https://github.com/sczhou/CodeFormer/blob/master/inputs/gray_faces/Hepburn02.png?raw=true")

# Iterate over outputs
for output in algo.get_outputs():
    # Print information
    print(output)
    # Export it to JSON
    output.to_json()
```


