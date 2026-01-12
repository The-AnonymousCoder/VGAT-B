from osgeo import ogr
import os

'''Read vector data (shp)'''


def Read_XYPo_fromshp(ori_shp):
    ds = ogr.Open(ori_shp, 0)
    layer = ds.GetLayer(0)
    feature_num = layer.GetFeatureCount()
    XLst, YLst, PoLst, X_sum, Y_sum = [], [], [], [], []
    for in_feature in layer:
        geom = in_feature.geometry()
        wkt = geom.ExportToWkt()
        pointstring = wkt[wkt.find('(') + 1:wkt.rfind(')')].split(',')
        x, y, po = [], [], []
        for point_value in pointstring:
            xy = point_value.strip().split(' ')
            if len(xy) >= 2:
                try:
                    x_val = float(xy[0])
                    y_val = float(xy[1])
                    x.append(x_val)
                    y.append(y_val)
                    po.append((x_val, y_val))
                except ValueError:
                    print(f"跳过无效坐标: {point_value}")
                    continue
        X_sum += x  
        Y_sum += y
        XLst.append(x), YLst.append(y), PoLst.append(po)
    ds.Destroy()
    return XLst, YLst, PoLst, feature_num, X_sum, Y_sum


'''Write out vector map data'''


def Write_XYPo_toshp(ori_shp, en_shp, XLst_emb, YLst_emb):
    ds = ogr.Open(ori_shp, 0)
    in_layer = ds.GetLayer(0)
    feature_num = in_layer.GetFeatureCount()
    driver = ogr.GetDriverByName('ESRI Shapefile')
    data_source = driver.CreateDataSource(en_shp)
    get_srs = in_layer.GetSpatialRef()
    out_layer = data_source.CreateLayer(en_shp, get_srs, ogr.wkbLineString)
    out_layer.CreateFields(in_layer.schema)
    out_defn = out_layer.GetLayerDefn()
    if os.access(en_shp, os.F_OK):
        driver.DeleteDataSource(en_shp)
    feat_count = 0
    for in_feature in in_layer:
        geom = in_feature.geometry()
        wkt = geom.ExportToWkt()
        pointstring = wkt[wkt.find('(') + 1:wkt.rfind(')')].split(',')
        pts_new = ''
        point_num = 0
        for point_value in pointstring:
            xy = point_value.split(' ')
            x = float(xy[0])
            y = float(xy[1])
            en_x = XLst_emb[feat_count][point_num]
            en_y = YLst_emb[feat_count][point_num]
            pts_new += str(en_x) + ' ' + str(en_y) + ','
            point_num += 1
        en_wkt = wkt[0:wkt.find('(') + 1] + pts_new[:-1] + ')'
        out_feature = ogr.Feature(out_defn)
        en_Geometry = ogr.CreateGeometryFromWkt(en_wkt)
        out_feature.SetGeometry(en_Geometry)
        for i in range(1, in_feature.GetFieldCount()):
            value = in_feature.GetField(i)
            out_feature.SetField(i, value)
        out_layer.CreateFeature(out_feature)
        feat_count += 1
    ds.Destroy()


"""Scrambling based on Tent mapping"""


def Tent(feature_num, XList, YList, Xor, Yor, a, t0):
    for i in range(0, feature_num):
        X0 = Xor
        N = len(XList[i])
        Lx, RcLx = [], []

        for k in range(0, t0 + N):
            if X0 < 0.5:
                X0 = a * X0
            else:
                X0 = a * (1 - X0)
            Lx.append(X0)
        cLx = Lx[t0:]
        for j in cLx:
            xi = int(N * j) % N
            RcLx.append(xi)
        #  X coordinate encryption
        for j in range(0, len(XList[i])):
            XList[i][j], XList[i][RcLx[j]] = XList[i][RcLx[j]], XList[i][j]
    RXLst = XList

    for m in range(0, feature_num):
        Y0 = Yor
        N = len(YList[m])
        Ly, RcLy = [], []
        for s in range(0, t0 + N):
            if Y0 < 0.5:
                Y0 = a * Y0
            else:
                Y0 = a * (1 - Y0)
            Ly.append(Y0)
        cLy = Ly[t0:]
        for n in cLy:
            yi = int(N * n) % N
            RcLy.append(yi)
        #  Y coordinate encryption
        for n in range(0, len(YList[m])):
            YList[m][n], YList[m][RcLy[n]] = YList[m][RcLy[n]], YList[m][n]
    RYLst = YList

    return RXLst, RYLst


"""Scrambling based on PWLCM mapping"""


def PWLCM(feature_num, XList, YList, x0, B, t0):
    for i in range(0, feature_num):
        N = len(XList[i])
        xi = x0
        Li, RcLi = [], []
        for s in range(0, t0 + N):
            if 0 <= xi < B:
                xi = xi / B
            elif B <= xi < 0.5:
                xi = (xi - B) / (0.5 - B)
            elif 0.5 <= xi < 1 - B:
                xi = (1 - B - xi) / (0.5 - B)
            else:
                xi = (1 - xi) / B
            Li.append(xi)
        cLi = Li[t0:]  
        for j in cLi:
            x = int(N * j) % N
            RcLi.append(x)  
        for j in range(0, len(XList[i])):
            XList[i][j], XList[i][RcLi[j]] = XList[i][RcLi[j]], XList[i][j]
        for n in range(0, len(YList[i])):
            YList[i][n], YList[i][RcLi[n]] = YList[i][RcLi[n]], YList[i][n]
    return XList, YList


if __name__ == '__main__':
    # 使用与Zero_watermarking.py相同的读取方法
    from Zero_watermarking import Read_Shapfile
    fn_r = 'pso_data/Boundary.shp'
    XLst, YLst, feature_num = Read_Shapfile(fn_r)  # read vector data
    # parameter
    Xor, Yor = 0.45, 0.55  
    a, t0 = 1.5, 1000  
    x0, B, = 0.45, 0.25
    RXLst, RYLst = Tent(feature_num, XLst, YLst, Xor, Yor, a, t0)
    RXLst2, RYLst2 = PWLCM(feature_num, RXLst, RYLst, x0, B, t0)
    fn_w = "encrypted_boundary.shp"
    
    # 使用Read_Write.py中的写出函数
    from Read_Write import write_encrytpion_shp
    write_encrytpion_shp(fn_r, fn_w, RXLst2, RYLst2)
    print("矢量地图加密完成！")
    print(f"加密了 {feature_num} 个特征")
    print(f"输出文件: {fn_w}")
