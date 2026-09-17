CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* **Name:** Yingxuan Hu
* **LinkedIn:** [linkedin.com/in/yingxuan-hu-bbb9b3380](https://www.linkedin.com/in/yingxuan-hu-bbb9b3380/)
* **Tested on:** Windows 11, Intel(R) Core(TM) i9-14900HX @ 2.20 GHz, NVIDIA GeForce RTX 5070 Ti Laptop GPU (12 GB)
* **Computer:** Personal Computer
* **Compute Capability:** 12.0 (`sm_120`)

## Implementation

This project implements several versions of exclusive prefix sum and stream compaction:

* **CPU Scan:** Serial exclusive scan using a simple loop.
* **Naive GPU Scan:** Parallel scan using multiple passes with increasing offsets and two alternating device buffers.
* **Work-Efficient GPU Scan:** Uses up-sweep and down-sweep passes and supports non-power-of-two input sizes through padding.
* **GPU Stream Compaction:** Uses map, scan, and scatter to remove zero values from the input array.
* **Thrust Scan:** Uses `thrust::exclusive_scan` for comparison.

## Performance Analysis

Performance tests were conducted in **Release mode** without debugging. Initial and final memory operations were excluded from GPU timing.

### Scan Performance vs. Array Size

![](images/result.png)

For small arrays, the CPU scan was much faster than the GPU implementations because the amount of work was too small to offset GPU kernel launch overhead.

As the array size increased, the CPU execution time increased more noticeably, while the GPU implementations became more competitive. At the largest tested size, the CPU, naive GPU, work-efficient GPU, and Thrust implementations were all in a similar range.

The naive GPU scan was faster than the work-efficient version for most of the tested sizes. Although the work-efficient algorithm performs less total work, it requires separate kernel launches for each level of the up-sweep and down-sweep. At higher levels of the tree, only a small number of threads are active, which reduces GPU utilization.

The work-efficient implementation became more competitive at the largest array size. This matches the idea that reducing total work becomes more useful as the amount of data increases.

The Thrust implementation was generally competitive with the custom GPU implementations. Its timing varied somewhat between different array sizes, but overall it showed the performance of an optimized library implementation compared with the simple scan implementations used in this project.

### Performance Bottlenecks

The CPU implementation mainly depends on the number of elements because it performs a simple linear scan.

The naive GPU implementation performs multiple full-array passes, so it requires more global memory accesses and more total work.

The work-efficient implementation reduces the total amount of work, but its performance is affected by repeated kernel launches and low thread utilization near the top of the up-sweep and down-sweep tree.

Overall, the results show that lower theoretical work does not always directly produce better GPU performance. Kernel launch overhead, memory access, thread utilization, and input size all affect the final execution time.

## Test Output

```text
****************
** SCAN TESTS **
****************
    [   4  36   0   4  17   0   0  16  41  24  32  40  28 ...  21   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 0.8523ms    (std::chrono Measured)
    [   0   4  40  40  44  61  61  61  77 118 142 174 214 ... 25699712 25699733 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.6004ms    (std::chrono Measured)
    [   0   4  40  40  44  61  61  61  77 118 142 174 214 ... 25699645 25699667 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 0.928352ms    (CUDA Measured)
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 0.376832ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 0.843776ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.587104ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 0.807488ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.45072ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   1   0   2   2   0   1   1   0   0   2   2   2   1 ...   1   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 3.6008ms    (std::chrono Measured)
    [   1   2   2   1   1   2   2   2   1   3   3   1   3 ...   1   1 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 3.5369ms    (std::chrono Measured)
    [   1   2   2   1   1   2   2   2   1   3   3   1   3 ...   2   1 ]
    passed
==== cpu compact with scan ====
   elapsed time: 1.5899ms    (std::chrono Measured)
    [   1   2   2   1   1   2   2   2   1   3   3   1   3 ...   1   1 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 0.660448ms    (CUDA Measured)
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.470176ms    (CUDA Measured)
    passed
```

## Build Note

I modified `CMakeLists.txt` to add `/Zc:preprocessor` for CUDA compilation on Windows. This was required for compatibility with CUDA 13.3 and the current Thrust/CCCL headers.
