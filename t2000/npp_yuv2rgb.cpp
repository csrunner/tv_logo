#include <NPPTool.h>
#include <npp.h>
#include <stdio.h>
//YUV420 TO RGB
void YUV2RGB(unsigned char* src_mem_y, unsigned char* src_mem_uv, unsigned char* dst_mem, int width,
        int height)
{       
        //all image step/size variables
        int rSrcStep[3];
        rSrcStep[0] = width;
        rSrcStep[1] = width/2;
        rSrcStep[2] = width/2;
        int Pixels = width * height;
        
        int dst_step = width;
        
        Npp8u* pSrc[3];
        
        NppiSize roi;
    roi.width  = width;
    roi.height = height;
        
        //step 1 : YUV420 TO RGB
        pSrc[0] = (Npp8u*)src_mem_y;
        pSrc[1] = (Npp8u*)src_mem_uv;
        pSrc[2] = (Npp8u*)src_mem_uv + Pixels / 4;
        
        Npp8u* pDst[3];
        pDst[0] = (Npp8u*)dst_mem;
        pDst[1] = (Npp8u*)dst_mem + Pixels;
        pDst[2] = (Npp8u*)dst_mem + Pixels * 2;
        nppiYUV420ToRGB_8u_P3R(pSrc, rSrcStep, pDst, dst_step, roi);

        return;
}
