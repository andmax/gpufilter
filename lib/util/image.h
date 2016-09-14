/**
 *  @file image.h
 *  @brief Image and pixel utility functions definition
 *  @author Rodolfo Lima
 *  @author Andre Maximo
 *  @date December, 2011
 *  @copyright The MIT License
 */

#ifndef IMAGE_H
#define IMAGE_H

//== NAMESPACES ================================================================

namespace gpufilter {

//== DEFINITIONS ===============================================================

/**
 *  @ingroup utils
 *  @brief Enumerates possible initial conditions (border type)
 *
 *  Border type is one out of four: clamp to zero is zero (or no)
 *  padding; clamp to edge is constant padding extensions; repeat is
 *  periodic extension; reflect is even-periodic extension.
 */
enum BorderType
{
    CLAMP_TO_ZERO,
    CLAMP_TO_EDGE,
    REPEAT,
    REFLECT
};

/**
 *  @ingroup utils
 *  @brief Calculates an adjusted texture-based coordinate
 *
 *  Given a texture-based coordinate (from 0 to 1 representing pixel 0
 *  to the last pixel in height or width) and a border type, return an
 *  equivalent adjusted texture-based coordinate in the image space.
 *
 *  @param[in] t The input texture-based coordinate
 *  @param[in] border_type The border type
 *  @return The adjusted texture-based coordinate
 */
float adjcoord(float t, BorderType border_type);

/**
 *  @ingroup utils
 *  @brief Calculates the 1D index of a pixel considering borders
 *
 *  Given a pixel-based 2D index (from pixel 0 to the last pixel in
 *  height or width), the input 2D image width and height, and a
 *  border type, return the 1D index of the pixel in the image space
 *  considering the 2D image stored in a 1D array.
 *
 *  @param[in] x The x coordinate of the pixel (can be outside the image)
 *  @param[in] y The y coordinate of the pixel (can be outside the image)
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] border_type The border type
 *  @return The 1D pixel index for the image array
 */
int calcidx(int x, int y, int w, int h, BorderType border_type);

/**
 *  @ingroup utils
 *  @brief Gets the pixel value at a give pixel-based coordinate
 *
 *  Given an input 2D image, its width and height, a pixel-based 2D
 *  index (from pixel 0 to the last pixel in height or width), and a
 *  border type, return the value of the pixel at that index.
 *
 *  @param[in] data The input 2D image data
 *  @param[in] x The x coordinate of the pixel (can be outside the image)
 *  @param[in] y The y coordinate of the pixel (can be outside the image)
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] border_type The border type
 *  @return The value of the pixel at x and y
 */
float getpix(const float *data, int x, int y, int w, int h, 
             BorderType border_type);

/**
 *  @ingroup utils
 *  @brief Extend an image to consider initial condition outside
 *
 *  Given an input 2D image extend it including a given initial
 *  condition for outside access.
 *
 *  @param[in] img The input 2D image to extend
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] border_top Number of pixels in the border top
 *  @param[in] border_left Number of pixels in the border left
 *  @param[in] border_bottom Number of pixels in the border bottom
 *  @param[in] border_right Number of pixels in the border right
 *  @param[in] border_type The border type
 *  @return The extended 2D image (allocated inside this function)
 */
float *extend_image(const float *img, int w, int h, 
                    int border_top, int border_left,
                    int border_bottom, int border_right,
                    BorderType border_type);

/**
 *  @ingroup utils
 *  @brief Calculates the number of pixels in borders
 *
 *  Given the width, height and the number of blocks in the border,
 *  this functions calculates the number of pixels in each direction
 *  outside the input 2D image.  Each block has 32x32 pixels.  This
 *  function also increments the number of pixels in borders to ensure
 *  the extended image size is a multiple of blocks.
 *
 *  @param[out] left Number of pixels in the border left
 *  @param[out] top Number of pixels in the border top
 *  @param[out] right Number of pixels in the border right
 *  @param[out] bottom Number of pixels in the border bottom
 *  @param[in] w Width of the input image
 *  @param[in] h Height of the input image
 *  @param[in] border Number of blocks (32x32) in the border
 */
void calc_borders(int *left, int *top, int *right, int *bottom,
                  int w, int h, int border=0);

/**
 *  @ingroup utils
 *  @brief Crop an image from an extended image
 *
 *  Given an input 2D extended image (with a number of pixels in each
 *  border direction) extract the original image in the middle.
 *
 *  @param[out] cimg The cropped 2D image
 *  @param[in] img The extended 2D image
 *  @param[in] w Width of the cropped image
 *  @param[in] h Height of the cropped image
 *  @param[in] border_top Number of pixels in the border top
 *  @param[in] border_left Number of pixels in the border left
 *  @param[in] border_bottom Number of pixels in the border bottom
 *  @param[in] border_right Number of pixels in the border right
 */
void crop_image(float *cimg, const float *img, int w, int h, 
                int border_top, int border_left,
                int border_bottom, int border_right);

//==============================================================================
} // namespace gpufilter
//==============================================================================
#endif // IMAGE_H
//==============================================================================
