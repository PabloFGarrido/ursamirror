# UrsaMirror

A Python package and GUI for analyzing the Mirror Tracing Task in polar coordinates.

[![PyPI - Version](https://img.shields.io/pypi/v/ursamirror.svg)](https://pypi.org/project/ursamirror)
[![DOI](https://zenodo.org/badge/873064551.svg)](https://doi.org/10.5281/zenodo.13987806)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ursamirror.svg)](https://pypi.org/project/ursamirror)
[![Contributor Covenant](https://img.shields.io/badge/Contributor%20Covenant-2.1-4baaaa.svg)](code_of_conduct.md) 
-----

## Table of Contents
- [Highlights](#highlights)
- [Usage](#usage)
- [Installation](#installation)
- [License](#license)
- [How to cite](#how-to-cite)

## Highlights

- Analyze the residuals and density of the drawing for every angle.
- Save this values in a .csv file or work with them as a pandas DataFrame.
- Transform the digitized paper drawing into and easy-to-work format.
- Transform the results from [`this online version`](https://neuron.illinois.edu/games/mirror-tracing-game/index.html?shape=star5)  of the drawing task into and easy-to-work format.
- Reconstruct the drawn path when it crosses the borders of the star.

## Usage
UrsaMirror performs all of its analysis using a standardized image format where the red, green and blue channels contain, respectively, the drawn path, the inner part of the star, and the borders of the star. You can simply transform any digitized image into this format using either `paper2std()` or `online2std()` functions. With the transformed image, the basic analysis in polar coordinates can be done in few lines of code.

There are several provided examples that you can use to test the package. They are located in the [`raw_data`](/raw_data) folder, and are divided into app, paper and online source. The app drawings come from an application developed in [`our group`](https://www.lcbc.uio.no/english/), whichs uses the standardize format as output. The paper folder contains digitized drawings. The online folder contains images from [`this online version of the task`](https://neuron.illinois.edu/games/mirror-tracing-game/index.html?shape=star5).

### Paper to standard version
Using one of the provided digitized paper examples, we can transform it into the standardized version:

```python
import ursamirror as um

path_to_image = "/raw_data/paper/blue.png"

transformed_image = um.paper2std(path_to_image)
```
You can now access and visualize each of the channels:

```python
import matplotlib.pylab as plt

fig, ax = plt.subplots(1,4, sharex=True,sharey=True)

ax[0].imshow(transformed_image[:,:,0], cmap="Reds")
ax[0].set_title("Path", fontsize=15)
ax[0].axis('off')

ax[1].imshow(transformed_image[:,:,1], cmap="Greens")
ax[1].set_title("Inner star", fontsize=15)
ax[1].axis('off')

ax[2].imshow(transformed_image[:,:,2], cmap="Blues")
ax[2].set_title("Borders", fontsize=15)
ax[2].axis('off')

ax[3].imshow(transformed_image)
ax[3].set_title("Transformed image", fontsize=15)
ax[3].axis('off')

plt.show()
```

Or you can just save it in the new format to be analyzed. 

```python
import ursamirror as um

path_to_image = "/raw_data/paper/blue.png"
new_path = "/processed_data/transformed_blue.png"

transformed_image = um.paper2std(path_to_image, new_path, save=True)
```

If you want to process a group of images, it can be done in a for loop. Check this simple example

```python
import ursamirror as um
import os 

path_to_folder = "/raw_data/paper/"
path_to_new_folder = "/processed_data/"

for file in os.listdir(path_to_folder):
    if ".png" in file:
        image_name = file
        #In Windows, change "/" by "\"
        path_to_image = path_to_folder + "/" + image_name

        new_name = "transformed_" + image_name
        new_path = "/processed_data/" + "/" + new_name

        transformed_image = um.paper2std(path_to_image, new_path, save=True)
```

### Online to standard version
The procedure is the same as the above. The only change here is the function to be used:

```python
import ursamirror as um

path_to_image = "/raw_data/online/00_online.png"

transformed_image = um.online2std(path_to_image)

#or saving the image
new_path = "/processed_data/transformed_00_online.png"
transformed_image = um.online2std(path_to_image, new_path, save=True)
```

### Star analysis

Once you have a star in the standardized format, the basic analysis and export can be done as:

```python
import ursamirror as um

path_to_image = "/processed_data/transformed_blue.png"

star = um.STAR(path_to_image)
export_data = star.export()
export_data.to_csv("/processed_data/transformed_blue.csv", index=False)
```

"transformed_blue.csv" will now contain 5 columns: The angles (in radians), the angles (in degrees), the density, the residuals, and the squared residuals. 

You can also pre-visualize the results
```python
star = um.STAR(path_to_image)
star.plot()
```

There are some other methods and attributes connected to the STAR object. The basic usage of the package are summarized above, but you can also get:

- Access to the path, inner star, and border, as from the standardized format.
- The skeleton of the path
- The inner/outer borders and their fitted coefficients.
- The center coordinates of the star in the image
- The estimated path thickness of the star.
- The drawn path and its skeleton in polar coordinates
- The residuals
- The density


## Installation

```console
pip install ursamirror
```
```console
pip install ursamirror@git+https://github.com/PabloFGarrido/ursamirror
```
If you are not comfortable working with Python or programming, try the GUI that you can find in the [`exec`](/exec) folder. 

Windows:
Download ursamiror_windows and run it.

MacOS and Linux: Download ursamiror_linux and run it from the terminal. From the same directory where the file is located, 

```console
./ursamiror_linux
```

## License

`ursamirror` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.

## How to cite
