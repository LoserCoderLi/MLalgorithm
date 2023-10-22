import numpy as np

data = np.random.normal(0, 1, [297, 13] )

np.random.seed(10)
dataA = np.random.normal(0, 1, [297, 13] )
# dataA = np.array(dataA)
y1 = np.random.normal(0, 1, [297, 1])

np.random.seed(11)
dataB = np.random.normal(0, 1, [297, 13])
y2 = np.random.normal(0, 1, [297, 1])

np.random.seed(12)
dataC = np.random.normal(0, 1, [297, 13])
y3 = np.random.normal(0, 1, [297, 1])

upnum = dataA.shape[1]
numlap = {'lap_ab':0,
          'lap_ac':0,
          'lap_bc':0,
          'lap_abc':0}
#重叠率是遍历
lap_a_b_c = {}

for lap in range(0,upnum):
    # print(lap)

#AB\AC\BC两个相互干扰情况1
    for i,i1 in zip(range(0,lap),range((upnum-lap),upnum)):

        #AB模拟重叠
        dataAcopy = dataA
        dataBcopy = dataB

        for j in range(0,data.shape[0]):

            dataAcopy[j][i1] += dataB[j][i]
            dataBcopy[j][i] += dataA[j][i1]

            #神经网络传递预测
            #略
            #预测值为y_preA,y_preB
    y_preA = np.random.normal(0,1,1)
    y_preB = np.random.normal(0,1,1)

    Aloss__ = (y1 - y_preA).var()
    Bloss__ = (y2 - y_preB).var()

            #判断
    if Aloss__>0.4 or Bloss__>0.4 :
        if numlap['lap_ab'] < 2:
            numlap['lap_ab'] += 1
            lap_a_b_c['lapab0'] = lap

# ----------------------------------------------------------------------------
    for i, i1 in zip(range(0, lap), range((upnum - lap), upnum)):

        #AC模拟重叠
        dataAcopy = dataA
        dataCcopy = dataC

        for j in range(0, data.shape[0]):
            dataAcopy[j][i1] += dataC[j][i]
            dataCcopy[j][i] += dataA[j][i1]

            # 神经网络传递预测
            # 略
            # 预测值为y_preA,y_preB(在此我先随便定义)
    y_preA = np.random.normal(0, 1, 1)
    y_preC = np.random.normal(0, 1, 1)

    Aloss__ = (y1 - y_preA).var()
    Closs__ = (y3 - y_preC).var()

    # 判断
    if Aloss__ > 0.4 or Closs__ > 0.4:
        if numlap['lap_ac'] < 2:
            numlap['lap_ac'] += 1
            lap_a_b_c['lapac0'] = lap
#-------------------------------------------------------------------------
    for i, i1 in zip(range(0, lap), range((upnum - lap), upnum)):

        #BC模拟重叠
        dataBcopy = dataB
        dataCcopy = dataC

        for j in range(0, data.shape[0]):
            dataBcopy[j][i1] += dataC[j][i]
            dataCcopy[j][0] += dataB[j][i1]

            # 神经网络传递预测向前传递
            # 略
            # 预测值为y_preA,y_preB
    y_preB = np.random.normal(0, 1, 1)
    y_preC = np.random.normal(0, 1, 1)

    Bloss__ = (y2 - y_preB).var()
    Closs__ = (y3 - y_preC).var()

    # 判断
    if Bloss__ > 0.4 or Closs__ > 0.4:
        if numlap['lap_bc'] < 2:
            numlap['lap_bc'] += 1
            lap_a_b_c['lapbc0'] = lap


#————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————

#AB\AC\BC两个相互干扰情况2
    for i, i1 in zip(range(0, lap), range((upnum - lap), upnum)):
        dataAcopy = dataA
        dataBcopy = dataB

        #AB模拟重叠
        for j in range(0, dataA.shape[0]):
            dataBcopy[j][i1] += dataA[j][i]
            dataAcopy[j][i] += dataB[j][i1]

        # 神经网络传递预测
        #略
        # 预测值为y_preA,y_preB
    y_preA = np.random.normal(0, 1, 1)
    y_preB = np.random.normal(0, 1, 1)

    Aloss__ = (y1 - y_preA).var()
    Bloss__ = (y2 - y_preB).var()
        #判断
    if Aloss__ > 0.4 or Bloss__ > 0.4:
        if numlap['lap_ab'] < 2:
            numlap['lap_ab'] += 1
            lap_a_b_c['lapab1'] = lap
#---------------------------------------------------------------------
    for i, i1 in zip(range(0, lap), range((upnum - lap), upnum)):

        #AC模拟重叠
        dataAcopy = dataA
        dataCcopy = dataC

        for j in range(0, data.shape[0]):
            dataCcopy[j][i1] += dataA[j][i]
            dataAcopy[j][i] += dataC[j][i1]

            # 神经网络传递预测
            # 略
            # 预测值为y_preA,y_preB
    y_preA = np.random.normal(0, 1, 1)
    y_preC = np.random.normal(0, 1, 1)

    Aloss__ = (y1 - y_preA).var()
    Closs__ = (y3 - y_preC).var()

    # 判断
    if Aloss__ > 0.4 or Closs__ > 0.4:
        if numlap['lap_ac'] < 2:
            numlap['lap_ac'] += 1
            lap_a_b_c['lapac1'] = lap
#----------------------------------------------------------------------------
    for i, i1 in zip(range(0, lap), range((upnum - lap), upnum)):

        #BC模拟重叠
        dataBcopy = dataB
        dataCcopy = dataC

        for j in range(0, data.shape[0]):
            dataCcopy[j][i1] += dataB[j][i]
            dataBcopy[j][i] += dataC[j][i1]

            # 神经网络传递预测向前传递
            # 略
            # 预测值为y_preA,y_preB
    y_preB = np.random.normal(0, 1, 1)
    y_preC = np.random.normal(0, 1, 1)

    Bloss__ = (y2 - y_preB).var()
    Closs__ = (y3 - y_preC).var()

    # 判断
    if Bloss__ > 0.4 or Closs__ > 0.4:
        if numlap['lap_bc'] < 2:
            numlap['lap_bc'] += 1
            lap_a_b_c['lapbc1'] = lap

#——————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————————
    for i, i1 in zip(range(0, lap), range((upnum - lap), upnum)):

        # ABC都重叠
        dataAcopy = dataA
        dataBcopy = dataB
        dataCcopy = dataC
        translate = np.random.randint(0, (upnum - lap))

        for j in range(0, data.shape[0]):
            dataAcopy[j][i1] += dataB[j][i]+dataC[j][i1-translate]
            dataBcopy[j][i] += dataA[j][i1]+dataC[j][i1-translate]
            dataCcopy[j][i1-translate] = dataB[j][i]+dataA[j][i1]

            # 神经网络传递预测
            # 略
            # 预测值为y_preA,y_preB
    y_preA = np.random.normal(0, 1, 1)
    y_preB = np.random.normal(0, 1, 1)
    y_preC = np.random.normal(0, 1, 1)
    Aloss__ = (y1 - y_preA).var()
    Bloss__ = (y2 - y_preB).var()
    Closs__ = (y3 - y_preC).var()
    # 判断
    if Aloss__ > 0.4 or Bloss__ > 0.4 or Closs__>0.4:
        if numlap['lap_abc'] < 1:
            numlap['lap_abc'] += 1
            lap_a_b_c['lapabc'] = lap




