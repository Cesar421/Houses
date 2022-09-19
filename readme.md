# House Prices - Advanced Regression Techniques

<img src = "Images/thumb76_76.png" width = 300 height = 200>

## Sections

- [House Prices - Advanced Regression Techniques](#house-prices---advanced-regression-techniques)
  - [Sections](#sections)
    - [Libraries and tools used](#libraries-and-tools-used)
    - [Data adquisition](#data-adquisition)
    - [Data exploration](#data-exploration)
    - [Data preproccesing](#data-preproccesing)
    - [Data Modeling](#data-modeling)
    - [Conclusions](#conclusions)


### Libraries and tools used

Here are some libraries used for this projects

```python
import numpy as np
import pandas as pd
import platform
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored
```

### Data adquisition

- The Data set was adquired via kaggle reposities on challenger chapter.
  - [kaggle.com](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques "House Prices - Advanced Regression Techniques data Set")
- It was stored in a local device using git for version control and sync via GitHub.
  - [GitHub_Repository](https://github.com/Cesar421/Houses "Cesar GitHub")
  
### Data exploration

Here we start the exploration of the data,

```python
if platform.system() == 'Windows':
    df_test = pd.read_csv("test.csv")
    df_train = pd.read_csv("train.csv")
elif platform.system() == 'Linux':
    df_test = pd.read_csv("test.csv")
    df_train = pd.read_csv("train.csv")

```

This is the info of train data set

|   _#_  | **_Column_** | **_Non-Null Count_** | **_Dtype_** |  **_#_**  | **_Column_** | **_Non-Null Count_** | **_Dtype_** |  **_#_**  |  **_Column_** | **_Non-Null Count_** | **_Dtype_** |
|:------:|:------------:|:--------------------:|:-----------:|:---------:|:------------:|:--------------------:|:-----------:|:---------:|:-------------:|:--------------------:|:-----------:|
|  _---_ | **_------_** | **_--------------_** | **_-----_** | **_---_** | **_------_** | **_--------------_** | **_-----_** | **_---_** |  **_------_** | **_--------------_** | **_-----_** |
|  **0** |      Id      |     1460 non-null    |    int64    |     31    |   BsmtCond   |     1423 non-null    |    object   |     62    |   GarageArea  |     1460 non-null    |    int64    |
|  **1** |  MSSubClass  |     1460 non-null    |    int64    |     32    | BsmtExposure |     1422 non-null    |    object   |     63    |   GarageQual  |     1379 non-null    |    object   |
|  **2** |   MSZoning   |     1460 non-null    |    object   |     33    | BsmtFinType1 |     1423 non-null    |    object   |     64    |   GarageCond  |     1379 non-null    |    object   |
|  **3** |  LotFrontage |     1201 non-null    |   float64   |     34    |  BsmtFinSF1  |     1460 non-null    |    int64    |     65    |   PavedDrive  |     1460 non-null    |    object   |
|  **4** |    LotArea   |     1460 non-null    |    int64    |     35    | BsmtFinType2 |     1422 non-null    |    object   |     66    |   WoodDeckSF  |     1460 non-null    |    int64    |
|  **5** |    Street    |     1460 non-null    |    object   |     36    |  BsmtFinSF2  |     1460 non-null    |    int64    |     67    |  OpenPorchSF  |     1460 non-null    |    int64    |
|  **6** |     Alley    |      91 non-null     |    object   |     37    |   BsmtUnfSF  |     1460 non-null    |    int64    |     68    | EnclosedPorch |     1460 non-null    |    int64    |
|  **7** |   LotShape   |     1460 non-null    |    object   |     38    |  TotalBsmtSF |     1460 non-null    |    int64    |     69    |   3SsnPorch   |     1460 non-null    |    int64    |
|  **8** |  LandContour |     1460 non-null    |    object   |     39    |    Heating   |     1460 non-null    |    object   |     70    |  ScreenPorch  |     1460 non-null    |    int64    |
|  **9** |   Utilities  |     1460 non-null    |    object   |     40    |   HeatingQC  |     1460 non-null    |    object   |     71    |    PoolArea   |     1460 non-null    |    int64    |
| **10** |   LotConfig  |     1460 non-null    |    object   |     41    |  CentralAir  |     1460 non-null    |    object   |     72    |     PoolQC    |      7 non-null      |    object   |
| **11** |   LandSlope  |     1460 non-null    |    object   |     42    |  Electrical  |     1459 non-null    |    object   |     73    |     Fence     |     281 non-null     |    object   |
| **12** | Neighborhood |     1460 non-null    |    object   |     43    |   1stFlrSF   |     1460 non-null    |    int64    |     74    |  MiscFeature  |      54 non-null     |    object   |
| **13** |  Condition1  |     1460 non-null    |    object   |     44    |   2ndFlrSF   |     1460 non-null    |    int64    |     75    |    MiscVal    |     1460 non-null    |    int64    |
| **14** |  Condition2  |     1460 non-null    |    object   |     45    | LowQualFinSF |     1460 non-null    |    int64    |     76    |     MoSold    |     1460 non-null    |    int64    |
| **15** |   BldgType   |     1460 non-null    |    object   |     46    |   GrLivArea  |     1460 non-null    |    int64    |     77    |     YrSold    |     1460 non-null    |    int64    |
| **16** |  HouseStyle  |     1460 non-null    |    object   |     47    | BsmtFullBath |     1460 non-null    |    int64    |     78    |    SaleType   |     1460 non-null    |    object   |
| **17** |  OverallQual |     1460 non-null    |    int64    |     48    | BsmtHalfBath |     1460 non-null    |    int64    |     79    | SaleCondition |     1460 non-null    |    object   |
| **18** |  OverallCond |     1460 non-null    |    int64    |     49    |   FullBath   |     1460 non-null    |    int64    |     80    |   SalePrice   |     1460 non-null    |    int64    |
| **19** |   YearBuilt  |     1460 non-null    |    int64    |     50    |   HalfBath   |     1460 non-null    |    int64    |           |               |                      |             |
| **20** | YearRemodAdd |     1460 non-null    |    int64    |     51    | BedroomAbvGr |     1460 non-null    |    int64    |           |               |                      |             |
| **21** |   RoofStyle  |     1460 non-null    |    object   |     52    | KitchenAbvGr |     1460 non-null    |    int64    |           |               |                      |             |
| **22** |   RoofMatl   |     1460 non-null    |    object   |     53    |  KitchenQual |     1460 non-null    |    object   |           |               |                      |             |
| **23** |  Exterior1st |     1460 non-null    |    object   |     54    | TotRmsAbvGrd |     1460 non-null    |    int64    |           |               |                      |             |
| **24** |  Exterior2nd |     1460 non-null    |    object   |     55    |  Functional  |     1460 non-null    |    object   |           |               |                      |             |
| **25** |  MasVnrType  |     1452 non-null    |    object   |     56    |  Fireplaces  |     1460 non-null    |    int64    |           |               |                      |             |
| **26** |  MasVnrArea  |     1452 non-null    |   float64   |     57    |  FireplaceQu |     770 non-null     |    object   |           |               |                      |             |
| **27** |   ExterQual  |     1460 non-null    |    object   |     58    |  GarageType  |     1379 non-null    |    object   |           |               |                      |             |
| **28** |   ExterCond  |     1460 non-null    |    object   |     59    |  GarageYrBlt |     1379 non-null    |   float64   |           |               |                      |             |
| **29** |  Foundation  |     1460 non-null    |    object   |     60    | GarageFinish |     1379 non-null    |    object   |           |               |                      |             |
| **30** |   BsmtQual   |     1423 non-null    |    object   |     61    |  GarageCars  |     1460 non-null    |    int64    |           |               |                      |             |

This is the info of test data set

|  **#**  |  **Column**  | **Non-Null Count** | **Dtype** |  **#**  |  **Column**  | **Non-Null Count** | **Dtype** |  **#**  |   **Column**  | **Non-Null Count** | **Dtype** |
|:-------:|:------------:|:------------------:|:---------:|:-------:|:------------:|:------------------:|:---------:|:-------:|:-------------:|:------------------:|:---------:|
| **---** |  **------**  | **--------------** | **-----** | **---** |  **------**  | **--------------** | **-----** | **---** |   **------**  | **--------------** | **-----** |
|  **0**  |      Id      |    1459 non-null   |   int64   |    31   |   BsmtCond   |    1414 non-null   |   object  |    62   |   GarageArea  |    1458 non-null   |  float64  |
|  **1**  |  MSSubClass  |    1459 non-null   |   int64   |    32   | BsmtExposure |    1415 non-null   |   object  |    63   |   GarageQual  |    1381 non-null   |   object  |
|  **2**  |   MSZoning   |    1455 non-null   |   object  |    33   | BsmtFinType1 |    1417 non-null   |   object  |    64   |   GarageCond  |    1381 non-null   |   object  |
|  **3**  |  LotFrontage |    1232 non-null   |  float64  |    34   |  BsmtFinSF1  |    1458 non-null   |  float64  |    65   |   PavedDrive  |    1459 non-null   |   object  |
|  **4**  |    LotArea   |    1459 non-null   |   int64   |    35   | BsmtFinType2 |    1417 non-null   |   object  |    66   |   WoodDeckSF  |    1459 non-null   |   int64   |
|  **5**  |    Street    |    1459 non-null   |   object  |    36   |  BsmtFinSF2  |    1458 non-null   |  float64  |    67   |  OpenPorchSF  |    1459 non-null   |   int64   |
|  **6**  |     Alley    |    107 non-null    |   object  |    37   |   BsmtUnfSF  |    1458 non-null   |  float64  |    68   | EnclosedPorch |    1459 non-null   |   int64   |
|  **7**  |   LotShape   |    1459 non-null   |   object  |    38   |  TotalBsmtSF |    1458 non-null   |  float64  |    69   |   3SsnPorch   |    1459 non-null   |   int64   |
|  **8**  |  LandContour |    1459 non-null   |   object  |    39   |    Heating   |    1459 non-null   |   object  |    70   |  ScreenPorch  |    1459 non-null   |   int64   |
|  **9**  |   Utilities  |    1457 non-null   |   object  |    40   |   HeatingQC  |    1459 non-null   |   object  |    71   |    PoolArea   |    1459 non-null   |   int64   |
|  **10** |   LotConfig  |    1459 non-null   |   object  |    41   |  CentralAir  |    1459 non-null   |   object  |    72   |     PoolQC    |     3 non-null     |   object  |
|  **11** |   LandSlope  |    1459 non-null   |   object  |    42   |  Electrical  |    1459 non-null   |   object  |    73   |     Fence     |    290 non-null    |   object  |
|  **12** | Neighborhood |    1459 non-null   |   object  |    43   |   1stFlrSF   |    1459 non-null   |   int64   |    74   |  MiscFeature  |     51 non-null    |   object  |
|  **13** |  Condition1  |    1459 non-null   |   object  |    44   |   2ndFlrSF   |    1459 non-null   |   int64   |    75   |    MiscVal    |    1459 non-null   |   int64   |
|  **14** |  Condition2  |    1459 non-null   |   object  |    45   | LowQualFinSF |    1459 non-null   |   int64   |    76   |     MoSold    |    1459 non-null   |   int64   |
|  **15** |   BldgType   |    1459 non-null   |   object  |    46   |   GrLivArea  |    1459 non-null   |   int64   |    77   |     YrSold    |    1459 non-null   |   int64   |
|  **16** |  HouseStyle  |    1459 non-null   |   object  |    47   | BsmtFullBath |    1457 non-null   |  float64  |    78   |    SaleType   |    1458 non-null   |   object  |
|  **17** |  OverallQual |    1459 non-null   |   int64   |    48   | BsmtHalfBath |    1457 non-null   |  float64  |    79   | SaleCondition |    1459 non-null   |   object  |
|  **18** |  OverallCond |    1459 non-null   |   int64   |    49   |   FullBath   |    1459 non-null   |   int64   |         |               |                    |           |
|  **19** |   YearBuilt  |    1459 non-null   |   int64   |    50   |   HalfBath   |    1459 non-null   |   int64   |         |               |                    |           |
|  **20** | YearRemodAdd |    1459 non-null   |   int64   |    51   | BedroomAbvGr |    1459 non-null   |   int64   |         |               |                    |           |
|  **21** |   RoofStyle  |    1459 non-null   |   object  |    52   | KitchenAbvGr |    1459 non-null   |   int64   |         |               |                    |           |
|  **22** |   RoofMatl   |    1459 non-null   |   object  |    53   |  KitchenQual |    1458 non-null   |   object  |         |               |                    |           |
|  **23** |  Exterior1st |    1458 non-null   |   object  |    54   | TotRmsAbvGrd |    1459 non-null   |   int64   |         |               |                    |           |
|  **24** |  Exterior2nd |    1458 non-null   |   object  |    55   |  Functional  |    1457 non-null   |   object  |         |               |                    |           |
|  **25** |  MasVnrType  |    1443 non-null   |   object  |    56   |  Fireplaces  |    1459 non-null   |   int64   |         |               |                    |           |
|  **26** |  MasVnrArea  |    1444 non-null   |  float64  |    57   |  FireplaceQu |    729 non-null    |   object  |         |               |                    |           |
|  **27** |   ExterQual  |    1459 non-null   |   object  |    58   |  GarageType  |    1383 non-null   |   object  |         |               |                    |           |
|  **28** |   ExterCond  |    1459 non-null   |   object  |    59   |  GarageYrBlt |    1381 non-null   |  float64  |         |               |                    |           |
|  **29** |  Foundation  |    1459 non-null   |   object  |    60   | GarageFinish |    1381 non-null   |   object  |         |               |                    |           |
|  **30** |   BsmtQual   |    1415 non-null   |   object  |    61   |  GarageCars  |    1458 non-null   |  float64  |         |               |                    |           |

### Data preproccesing

- #### Numerical variables
  
- #### Categorical variables

### Data Modeling

### Conclusions
