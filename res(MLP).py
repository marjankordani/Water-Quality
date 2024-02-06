import numpy as np
import pandas as pd
import keras.models as mod
import keras.layers as lay
import keras.losses as los
import keras.optimizers as opt
import keras.activations as act
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
from Tools import *

ResetRandom()
plt.style.use('ggplot')


Show = True
Save = False
DPI = 384


nClass = 10
TimeFrame = 5
nLag = 10
nFuture = 1
sTr = 60 / 100
sVa = 15 / 100
nDenses = [128]
lstmActivation = act.tanh
denseActivation = act.selu
outputActivation = act.linear
LR = 1e-3
B1 = 9e-1
Loss = los.MeanSquaredError()
sBatch = 128
nEpoch = 30

ModelName = __file__.split('\\')[-1][:-3]
ResultPath = f'Results/{ModelName}'

if Save:
    if not os.path.exists(ResultPath):
        os.makedirs(ResultPath)

DF = Fetch('Data')

xColumns = DF.columns.to_list()
yColumns = ['Chlorophyll', 'Oxygen']

for yColumn in yColumns:
    TNs = [yColumn]

    Sx00 = DF[TNs].to_numpy()
    Sy00 = DF[TNs].to_numpy()
    
    Sx0 = TimeFrame2(Sx00, TimeFrame)
    Sy0 = TimeFrame2(Sy00, TimeFrame)
    
    SSX = pp.StandardScaler()
    Sx = SSX.fit_transform(Sx0)

    SSY = pp.StandardScaler()
    Sy = SSY.fit_transform(Sy0)
    
    X, Y, fX = Lag(Sx,
                   Sy,
                   nLag,
                   nFuture)
    
    trX, vaX, teX, trY, vaY, teY = TVT(X,
                                       Y,
                                       sTr,
                                       sVa)
    
    trY0 = SSY.inverse_transform(trY)
    vaY0 = SSY.inverse_transform(vaY)
    teY0 = SSY.inverse_transform(teY)
    
    InputShape = trX.shape[1:]
    nY = trY.shape[1]
    
    I = lay.Input(shape=InputShape)

    O = lay.Flatten()(I)

    for i in nDenses:
        Ot = lay.Dense(units=i,
                       activation=denseActivation)(O)
        O = lay.concatenate([O, Ot])
    
    O = lay.Dense(units=nY,
                  activation=outputActivation)(O)
    
    Model = mod.Model(inputs=I, outputs=O)

    Model.compile(optimizer=opt.Adam(learning_rate=LR, beta_1=B1),
                  loss=Loss)
    
    History = Model.fit(x=trX,
                        y=trY,
                        batch_size=sBatch,
                        epochs=nEpoch,
                        validation_data=(vaX, vaY),
                        shuffle=True).history
    
    trP = Model.predict(trX, verbose=0)
    vaP = Model.predict(vaX, verbose=0)
    teP = Model.predict(teX, verbose=0)
    
    trP0 = SSY.inverse_transform(trP)
    vaP0 = SSY.inverse_transform(vaP)
    teP0 = SSY.inverse_transform(teP)
    
    Summary(Model)

    PlotModel(Model, Save, DPI, ResultPath)

    LossPlot(History, 'MSE', TNs, Show, Save, DPI, ResultPath)
    
    RegressionReport(trY0, trP0, nFuture, 'Train', TNs, Show, Save, ResultPath)
    RegressionReport(vaY0, vaP0, nFuture, 'Validation', TNs, Show, Save, ResultPath)
    RegressionReport(teY0, teP0, nFuture, 'Test', TNs, Show, Save, ResultPath)
    
    RegressionPlot(trY0, trP0, 'Train', TNs, Show, Save, DPI, ResultPath)
    RegressionPlot(vaY0, vaP0, 'Validation', TNs, Show, Save, DPI, ResultPath)
    RegressionPlot(teY0, teP0, 'Test', TNs, Show, Save, DPI, ResultPath)

    SeriesPlot(trY0, trP0, 'Train', TNs, Show, Save, DPI, ResultPath)
    SeriesPlot(vaY0, vaP0, 'Validation', TNs, Show, Save, DPI, ResultPath)
    SeriesPlot(teY0, teP0, 'Test', TNs, Show, Save, DPI, ResultPath)