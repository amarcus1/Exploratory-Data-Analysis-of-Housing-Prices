# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 07:56:50 2022

@author: amarcus1
"""

def map_ordinals(data):
    """
    maps ordinal features to numeric values
    
    Arguments:
        data -- Features before transformation (Pandas Dataframe) 
        
    Returns:
        data -- Features after transformation (Pandas Dataframe) 
    """  
    
    # LandSlope: Slope of property
    LandSlope = {}
    LandSlope['Gtl'] = 3 #'Gentle slope'
    LandSlope['Mod'] = 2 #'Moderate Slope'
    LandSlope['Sev'] = 1 #'Severe Slope'

    data.LandSlope = data.LandSlope.map(LandSlope)
             
    # ExterCond: Evaluates the present condition of the material on the exterior
    ExterCond = {}
    ExterCond['Ex'] = 5 #'Excellent'
    ExterCond['Gd'] = 4 #'Good'
    ExterCond['TA'] = 3 #'Average/Typical'
    ExterCond['Fa'] = 2 #'Fair'
    ExterCond['Po'] = 1 #'Poor'
    data.ExterCond = data.ExterCond.map(ExterCond)   
    data.ExterCond = data.ExterCond.fillna(0)
 
    # FireplaceQu: Fireplace quality
    data.FireplaceQu = data.FireplaceQu.map(ExterCond)    
    data.FireplaceQu = data.FireplaceQu.fillna(0)

    #HeatingQC: Heating quality and condition
    data.HeatingQC = data.HeatingQC.map(ExterCond)
    data.HeatingQC = data.HeatingQC.fillna(0)
    
    # GarageCond: Garage Conditionals
    data.GarageCond = data.GarageCond.map(ExterCond)
    data.GarageCond = data.GarageCond.fillna(0)
    
    # GarageQual: Garage quality
    data.GarageQual = data.GarageQual.map(ExterCond)
    data.GarageQual = data.GarageQual.fillna(0)
        
    # ExterQual: Evaluates the quality of the material on the exterior 
    ExterQual = {}
    ExterQual['Ex'] = 4 #'Excellent'
    ExterQual['Gd'] = 3 #'Good'
    ExterQual['TA'] = 2 #'Average/Typical'
    ExterQual['Fa'] = 1 #'Fair'

    data.ExterQual = data.ExterQual.map(ExterQual)
    data.ExterQual = data.ExterQual.fillna(0)

    # BsmtQual: Evaluates the height of the basement
    data.BsmtQual = data.BsmtQual.map(ExterQual)
    data.BsmtQual = data.BsmtQual.fillna(0)

    # KitchenQual: Kitchen quality
    data.KitchenQual = data.KitchenQual.map(ExterQual)
    data.KitchenQual = data.KitchenQual.fillna(0)

    # BsmtCond: Evaluates the general condition of the basement
    data.BsmtCond = data.BsmtCond.map(ExterQual)
    data.BsmtCond = data.BsmtCond.fillna(0)


    # PoolQC: Pool quality
    PoolQC = {}
    PoolQC['Ex'] = 3 #'Excellent'
    PoolQC['Gd'] = 2 #'Good'
    PoolQC['Fa'] = 1 #'Fair'    
    data.PoolQC = data.PoolQC.map(ExterQual)
    data.PoolQC = data.PoolQC.fillna(0)

    PavedDrive = {}
    PavedDrive['Y'] = 3 #'Paved'
    PavedDrive['P'] = 2 #'Partial Pavement'
    PavedDrive['N'] = 1 #'Dirt/Gravel'

    data.PavedDrive = data.PavedDrive.map(PavedDrive)

    # LotShape: General shape of property
    LotShape = {}
    LotShape['Reg'] = 4 #'Regular'
    LotShape['IR1'] = 3 #'Slightly irregular'
    LotShape['IR2'] = 2 #'Moderately Irregular'
    LotShape['IR3'] = 1 #'Irregular'

    data.LotShape = data.LotShape.map(LotShape)
    
    # BsmtExposure: Refers to walkout or garden level walls
    BsmtExposure = {}
    BsmtExposure['Gd'] = 4 #'Good Exposure'
    BsmtExposure['Av'] = 3 #'Average Exposure (split levels or foyers typically score average or above)'
    BsmtExposure['Mn'] = 2 #'Mimimum Exposure'
    BsmtExposure['No'] = 1 #'No Exposure'

    data.BsmtExposure = data.BsmtExposure.map(BsmtExposure)
    data.BsmtExposure = data.BsmtExposure.fillna(0)  #'No Basement'

    # BsmtFinType1: Rating of basement finished area
    BsmtFinType1 = {}
    BsmtFinType1['GLQ'] = 6 #'Good Living Quarters'
    BsmtFinType1['ALQ'] = 5 # 'Average Living Quarters'
    BsmtFinType1['BLQ'] = 4 # 'Below Average Living Quarters'
    BsmtFinType1['Rec'] = 3 # 'Average Rec Room'
    BsmtFinType1['LwQ'] = 2 # 'Low Quality'
    BsmtFinType1['Unf'] = 1 # 'Unfinshed'

    data.BsmtFinType1 = data.BsmtFinType1.map(BsmtFinType1)
    data.BsmtFinType1 = data.BsmtFinType1.fillna(0)  #'No Basement'

    # BsmtFinType2: Rating of basement finished area (if multiple types)
    data.BsmtFinType2 = data.BsmtFinType2.map(BsmtFinType1)
    data.BsmtFinType2 = data.BsmtFinType2.fillna(0)  #'No Basement'

    #CentralAir: Central air conditioning
    # Since with this transformatio as the same as binarize this feature
    CentralAir = {}
    CentralAir['N'] = 0
    CentralAir['Y'] = 1

    data.CentralAir = data.CentralAir.map(CentralAir)

    # GarageFinish: Interior finish of the garage
    GarageFinish = {}
    GarageFinish['Fin'] = 3 #'Finished'
    GarageFinish['RFn'] = 2 #'Rough Finished'
    GarageFinish['Unf'] = 1 #'Unfinished'
    
    data.GarageFinish = data.GarageFinish.map(GarageFinish)
    data.GarageFinish = data.GarageFinish.fillna(0)  #'No Garage'
    
    # Functiol: Home functionality
    Functiol = {}
    Functiol['Typ'] = 7   # Typical Functionality
    Functiol['Min1'] = 6  # Minor Deductions 1
    Functiol['Min2'] = 5  # Minor Deductions 2
    Functiol['Mod'] = 4   # Moderate Deductions
    Functiol['Maj1'] = 3  # Major Deductions 1
    Functiol['Maj2'] = 2  # Major Deductions 2
    Functiol['Sev'] = 1   # Severely Damaged
    Functiol['Sal'] = 0   # Salvage only

    data.Functiol = data.Functiol.map(Functiol)
    
    #Street: Type of road access to property
    # Since with this transformatio as the same as binarize this feature
    Street = {}
    Street['Grvl'] = 0 # Gravel 
    Street['Pave'] = 1 # Paved

    data.Street = data.Street.map(Street)


    # Fence: Fence quality
    Fence = {}
    Fence['GdPrv'] = 5 #'Good Privacy'
    Fence['MnPrv'] = 4 #'Minimum Privacy'
    Fence['GdWo'] = 3 #'Good Wood'
    Fence['MnWw'] = 2 #'Minimum Wood/Wire'

    data.Fence = data.Fence.map(Fence)
    data.Fence = data.Fence.fillna(1)  #'No Fence'
    #But No Fence has the higest median Sales Price. So I try to use it as categorical
            
    return data

