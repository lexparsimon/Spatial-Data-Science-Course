#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

def func(gdf):
    mat2 = np.zeros(shape=(2660, 2660), dtype=np.uint16)
    for trip in gdf['trips'].unique():
        df = gdf[gdf['trips']==trip]
        mat2[df.pindex.iloc[0], df.pindex.iloc[-1]] += 1
    return mat2


# In[ ]:




