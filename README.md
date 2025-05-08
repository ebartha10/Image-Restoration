# Image Restoration

This project implements various image restoration algorithms to fill in missing or damaged regions in images. It includes three main approaches: Criminisi's exemplar-based inpainting, bilinear interpolation, and bicubic interpolation.

## Features

### 1. Criminisi's Exemplar-based Inpainting
- Implements the Criminisi algorithm for exemplar-based inpainting
- Uses patch-based texture synthesis
- Preserves both structure and texture
- Requires a mask image to identify regions to be filled
- Best for large missing regions with complex textures

### 2. Bilinear Interpolation
- Simple and fast interpolation method
- Uses 2x2 neighborhood for pixel interpolation
- Good for small missing regions
- Works well with smooth gradients
- Automatically detects white regions to be filled

### 3. Bicubic Interpolation
- Higher quality interpolation using 4x4 neighborhood
- Better preserves image details
- More accurate than bilinear interpolation
- Automatically detects white regions to be filled
- Good for both small and medium-sized missing regions

## Requirements

- OpenCV 4.x
- C++17 or later
- CMake 3.10 or later

## Building the Project

1. Clone the repository:
```bash
git clone https://github.com/ebartha10/Image-Restoration.git
cd Image-Restoration
```

2. Create a build directory and build the project:
```bash
mkdir build
cd build
cmake ..
cmake --build .
```

## Usage

The program provides three main functions for image restoration:

### Criminisi's Inpainting
```cpp
runCriminisi("path/to/image.png", "path/to/mask.png");
```
- Requires both an input image and a mask image
- Mask should be binary (0 for holes, 255 for filled regions)
- Best for complex textures and structures

### Bilinear Interpolation
```cpp
runBilinear("path/to/image.bmp");
```
- Only requires the input image
- Automatically detects white regions to be filled
- Good for simple gradients and small regions

### Bicubic Interpolation
```cpp
runBicubic("path/to/image.bmp");
```
- Only requires the input image
- Automatically detects white regions to be filled
- Better quality than bilinear for complex regions

## Example

```cpp
int main() {
    // Run Criminisi's inpainting
    runCriminisi("images/nature_small.png", "images/nature_small_mask.png");
    
    // Run bilinear interpolation
    runBilinear("images/gradientBrush.bmp");
    
    // Run bicubic interpolation
    runBicubic("images/gradientBrush.bmp");
    
    waitKey();
    return 0;
}
```

## Output

- Criminisi's inpainting shows progress in real-time
- Bilinear and bicubic interpolation save results to:
  - `result_bilinear.bmp`
  - `result_bicubic.bmp`

## Algorithm Details

### Criminisi's Algorithm
1. Identifies the boundary of the hole
2. Computes priority for each boundary pixel
3. Finds the best matching patch from the source region
4. Transfers the patch to the target region
5. Updates confidence and continues until the hole is filled

### Bilinear Interpolation
1. Detects white regions in the image
2. For each white pixel, uses 2x2 neighborhood for interpolation
3. Iteratively fills regions until no more changes occur

### Bicubic Interpolation
1. Detects white regions in the image
2. Uses 4x4 neighborhood for higher quality interpolation
3. Falls back to bilinear interpolation if bicubic fails
4. Iteratively processes until completion

## Author

[Your Name]

