/**
 *  @file image.cpp
 *  @brief Image and pixel utility functions implementation
 *  @author Rodolfo Lima
 *  @author Andre Maximo
 *  @date December, 2011
 *  @copyright The MIT License
 */

//== INCLUDES ==================================================================

#include <cmath>
#include <iostream>
#include <algorithm>
#include <cassert>

#include "image.h"

//== NAMESPACES ================================================================

namespace gpufilter {

//== IMPLEMENTATION ============================================================

float adjcoord(float t, BorderType border_type) {
    switch(border_type)
    {
    case CLAMP_TO_ZERO:
        return t;
    case CLAMP_TO_EDGE:
        if(t < 0)
            return 0;
        else if(t > 1)
            return 1;
        else
            return t;
    case REPEAT:
        t = fmodf(t, 1);
        return t < 0 ? t+1 : t;
    case REFLECT:
        t = fabs(fmodf(t,2));
        return t > 1 ? 2 - t : t;
    default:
        assert(false);
        return 0;
    }
}

int calcidx(int x, int y, int w, int h, BorderType border_type) {
    x = std::min<int>(w-1,adjcoord((x+.5f)/w,border_type)*w);
    y = std::min<int>(h-1,adjcoord((y+.5f)/h,border_type)*h);

    int idx = y*w+x;
    assert(idx >= 0 && idx < w*h);
    return idx;
}

float getpix(const float *data, int x, int y, int w, int h, 
             BorderType border_type) {
    if(border_type == CLAMP_TO_ZERO && (x<0 || x>=w || y<0 || y>=h))
        return 0;
    else
    {
        int idx = calcidx(x,y,w,h,border_type);
        return data[idx];
    }
}

float *extend_image(const float *img, int w, int h, 
                    int border_top, int border_left,
                    int border_bottom, int border_right,
                    BorderType border_type) {
    int nw = w+border_left+border_right,
        nh = h+border_top+border_bottom;

    float *newimg = new float[nw*nh];

    for(int y=0; y<nh; ++y)
    {
        for(int x=0; x<nw; ++x)
        {
            newimg[y*nw + x] 
                = getpix(img, x-border_left, y-border_top, w, h, border_type);
        }
    }

    return newimg;
}

void calc_borders(int *left, int *top, int *right, int *bottom,
                  int w, int h, int border) {
    if(border > 0)
    {
        *left = border*32;
        *top = border*32;

        *right = border*32;
        if (w%32>0) *right += 32-w%32;

        *bottom = border*32;
        if (h%32>0) *bottom += 32-h%32;

    }
    else
    {
        *left = *top = 0;

        *right = 32-(w%32);
        if (*right == 32)
            *right = 0;

        *bottom = 32-(h%32);
        if (*bottom == 32)
            *bottom = 0;
    }
}

void crop_image(float *cimg, const float *img, int w, int h, 
                int border_top, int border_left,
                int border_bottom, int border_right) {
    int nw = w+border_left+border_right;

    img += border_top*nw + border_left;

    for(int y=0; y<h; ++y, img += nw, cimg += w)
        std::copy(img, img+w, cimg);
}

//==============================================================================
} // namespace gpufilter
//==============================================================================
