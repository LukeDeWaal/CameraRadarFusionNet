cimport cython
cimport numpy as np
import numpy as np

np.import_array()
from numpy cimport ndarray as arr
from cython.view cimport array as cvarray
from libcpp cimport bool
from cython.parallel import prange


def vertical_line(
    arr[float, ndim=1] P1,
    arr[float, ndim=1] P2,
    arr[float, ndim=3] img,
):
    cdef unsigned int imageH = img.shape[0];
    cdef unsigned int imageW = img.shape[1];

    cdef int P1_y = int(P1[1]);
    cdef int P2_y = int(P2[1]);
    cdef int dX = 0;
    cdef int dY = P2_y - P1_y;
    if dY == 0:
        dY = 1;
    cdef int dXa = np.abs(dX);
    cdef int dYa = np.abs(dY);

    cdef arr[float, ndim = 2] itbuffer = np.empty(
        shape=(np.maximum(dYa, dXa), 2), dtype=np.float32
    );
    itbuffer.fill(np.nan)

    itbuffer[:, 0] = int(P1[0])
    #print(P1_y, dYa, itbuffer.shape[0], itbuffer.shape[1], P1_y - 1, P1_y - dYa - 1)
    if P1_y > P2_y:
        # Obtain coordinates along the line using a form of Bresenham's algorithm
        itbuffer[:, 1] = np.arange(P1_y - 1, P1_y - dYa - 1, -1)
    else:
        itbuffer[:, 1] = np.arange(P1_y + 1, P1_y + dYa + 1)

    cdef arr[int, ndim=1] colX = itbuffer[:, 0].astype(np.int32)
    cdef arr[int, ndim=1] colY = itbuffer[:, 1].astype(np.int32)
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) &
                        (colX < imageW) & (colY < imageH)]

    return itbuffer


#image_data, radar_data, radar_xyz_endpoints, clear_radar=False
def radar2cam(
    arr[float, ndim=3] image_data,
    arr[float, ndim=2] radar_data,
    arr[float, ndim=2] radar_end,
    unsigned int clear_radar
):
    """
    """
    cdef int radar_meta_count = radar_data.shape[0] - 3;
    cdef int img_h = image_data.shape[0];
    cdef int img_w = image_data.shape[1];
    cdef int img_c = image_data.shape[2];
    cdef int N = radar_data.shape[1];
    #cdef np.ndarray[float, ndim=3] radar_extension = np.zeros((img_h, img_w, radar_meta_count), dtype=np.float32)
    cdef arr[float, ndim=3] image_plus = np.zeros(
        shape=(img_h, img_w, radar_meta_count + img_c), dtype=np.float32);
    image_plus[:,:,:img_c] = image_data;

    cdef arr[float, ndim = 2] proj_line;
    cdef int y;
    cdef int x;

    if clear_radar > 0:
        return image_plus;

    else:
        for i in range(N): #prange(N, nogil=False, num_threads=6):
            proj_line = vertical_line(
                radar_data[0:2, i], radar_end[0:2, i], image_data
            )
            for j in range(proj_line.shape[0]):
                y = np.int32(proj_line[j, 1])
                x = np.int32(proj_line[j, 0])

                # Check if pixel is already filled with radar data and overwrite if distance is less than the existing
                if not np.any(image_plus[y, x, img_c:]) or radar_data[-1, i] < image_plus[y, x, -1]:
                    image_plus[y, x, img_c:] = radar_data[3:, i]

        return image_plus;
