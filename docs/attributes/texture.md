### Texture Attributes

|       **Atribute**        | **Description** |  **Status**  | **CPU** | **GPU** | **Multi-CPU** | **Multi-GPU** |
|:-------------------------:|:---------------:|:------------:|:-------:|:-------:|:-------------:|:-------------:|
|      GLCM Contrast        |                 |     Ready    |    X    |    X    |       X       |       X       |
|    GLCM Dissimilarity     |                 |     Ready    |    X    |    X    |       X       |       X       |
|     GLCM Homogeneity      |                 |     Ready    |    X    |    X    |       X       |       X       |
|       GLCM Energy         |                 |     Ready    |    X    |    X    |       X       |       X       |
|     GLCM Correlation      |                 |     Ready    |    X    |    X    |       X       |       X       |
|        GLCM ASM           |                 |     Ready    |    X    |    X    |       X       |       X       |
|        GLCM Mean          |                 |   Unstable   |         |    X    |               |       X       |
|      GLCM Variance        |                 |   Unstable   |         |    X    |               |       X       |
|         LBP 2D            |                 | Experimental |    X    |         |       X       |               |
|     LBP Diagonal 3D       |                 |     Ready    |    X    |    X    |       X       |       X       |

#### Observations:

* The attributes *GLCM Mean* and *GLCM Variance* are not implemented into scikit image.
* For GPU, GLCM requires CuPy release higher than 10.6.
* The attribute *LBP 2D* is not available for CUDA because CuCIM does not have this feature implemented yet.
