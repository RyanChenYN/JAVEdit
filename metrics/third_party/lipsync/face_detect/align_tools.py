import math
from typing import List
import numpy as np
class Point:
    def __init__(self, x: float = 0, y: float = 0):
        self.x = x
        self.y = y

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

def float_2_point(points: List[float]) -> List[Point]:
    ret_points = list()
    for point_x, point_y in zip(points[0::2], points[1::2]):
        ret_points.append(Point(point_x, point_y))

    return ret_points

def point_2_float(points: List[Point]) -> List[float]:
    ret_points = list()
    for point in points:
        ret_points.append(point.x)
        ret_points.append(point.y)

    return ret_points

def points_117_158_256(points158: List[float], points117: List[float]) -> List[float]:

    M_PI = 3.1415926

    FOREHEAD_POINTS_NUM = 7
    NOSE_POINTS_NUM = 22
    PROFILE_POINTS_NUM = 41
    # define sparse(130) points num
    SPARSE_EYEBROW_POINTS_NUM = 8
    SPARSE_EYE_POINTS_NUM = 8
    SPARSE_MOUTH_POINTS_NUM = 22
    SPARSE_PUPIL_POINTS_NUM = 6
    # define dense(256) points num
    DENSE_EYEBROW_POINTS_NUM = 16
    DENSE_EYE_POINTS_NUM = 24
    DENSE_MOUTH_POINTS_NUM = 72
    DENSE_PUPIL_POINTS_NUM = 34
    SPARSE_POINTS_NUM = (
        (SPARSE_EYEBROW_POINTS_NUM + SPARSE_EYE_POINTS_NUM) * 2
        + NOSE_POINTS_NUM
        + SPARSE_MOUTH_POINTS_NUM
        + PROFILE_POINTS_NUM
        + FOREHEAD_POINTS_NUM
        + SPARSE_PUPIL_POINTS_NUM
    )

    def Merge(densePoints, sparsePoints):
        points = list()
        count = 0
        # eyes
        for i in range((DENSE_EYEBROW_POINTS_NUM + DENSE_EYE_POINTS_NUM) * 2):
            points.append(densePoints[count])
            count += 1

        # 从 sparse 中取 鼻子的点位
        sparse_nose_start = (SPARSE_EYEBROW_POINTS_NUM + SPARSE_EYE_POINTS_NUM) * 2
        for i in range(NOSE_POINTS_NUM):
            points.append(sparsePoints[sparse_nose_start + i])

        # mouth
        for i in range(DENSE_MOUTH_POINTS_NUM):
            points.append(densePoints[count])
            count += 1

        # 从 sparse 中取轮廓的点位
        spase_profile_start = (
            sparse_nose_start + NOSE_POINTS_NUM + SPARSE_MOUTH_POINTS_NUM
        )
        for i in range(PROFILE_POINTS_NUM):
            points.append(sparsePoints[spase_profile_start + i])

        # pupil:TODO
        for i in range(SPARSE_PUPIL_POINTS_NUM):
            points.append(densePoints[count])
            count += 1

        return points

    def Norm(point):
        return math.sqrt(point.x * point.x + point.y * point.y)

    def GetPoint(p1, p2, sintheta, costheta, ratio=1.0):
        point = Point()
        distance = Norm(p1 - p2) * ratio
        point.x = p1.x + 2 * distance * sintheta
        point.y = p1.y - 2 * distance * costheta
        return point

    def AddNoseCourterPoints(sparsePts, densePts):
        for i in range(NOSE_POINTS_NUM):
            densePts[80 + i] = sparsePts[30 + i]

        for i in PROFILE_POINTS_NUM:
            densePts[174 + i] = sparsePts[76 + i]

    def AddForeheadPointsSub(points, meixin, longAx, shortBy, angle):
        deg = 22.5 / 180 * M_PI

        for j in range(3, 0, -1):
            tempPoint = Point()
            tempPoint.x = meixin.x + longAx * math.cos(j * deg)
            tempPoint.y = meixin.y - shortBy * math.sin(j * deg)
            beforeR = tempPoint - meixin
            afterR = Point()
            afterR.x = math.cos(angle) * beforeR.x + math.sin(angle) * beforeR.y
            afterR.y = math.sin(-angle) * beforeR.x + math.cos(angle) * beforeR.y
            rotatedPoint = afterR + meixin
            points.append(rotatedPoint)

    def AddForeheadPoints(points):
        # ptLu, ptLd, ptChin, ptRd, ptRu, ptNose, ptMeixin
        if len(points) <= SPARSE_POINTS_NUM:
            ptLu = points[76]  # 轮廓左上点（左起始点）
            ptLd = points[79]  # 轮廓左下点
            ptChin = points[96]  # 轮廓下巴点
            ptRd = points[113]  # 轮廓右下点
            ptRu = points[116]  # 轮廓右上点（右起始点）
            ptNose = points[43]
            ptMeixin = points[36]
        else:
            ptLu = points[174]  # 轮廓左上点（左起始点）
            ptLd = points[177]  # 轮廓左下点
            ptChin = points[194]  # 轮廓下巴点
            ptRd = points[211]  # 轮廓右下点
            ptRu = points[214]  # 轮廓右上点（右起始点）
            ptNose = points[91]
            ptMeixin = points[84]

        a = ptMeixin.x - ptNose.x
        b = ptMeixin.y - ptNose.y
        c = math.sqrt(a * a + b * b)
        costheta = -b / c
        sintheta = a / c
        faceAngle = math.atan2(a, b) + M_PI

        leftTemple = GetPoint(ptLd, ptLu, sintheta, costheta)
        rightTemple = GetPoint(ptRd, ptRu, sintheta, costheta)
        middleTemple = GetPoint(ptNose, ptMeixin, sintheta, costheta, 1.3)

        shortBy = Norm(middleTemple - ptMeixin)
        longAxRight = Norm(rightTemple - ptMeixin)
        longAxLeft = -(Norm(leftTemple - ptMeixin))

        # 额头右半边中间三点
        AddForeheadPointsSub(points, ptMeixin, longAxRight, shortBy, faceAngle)
        # 右半边逆序逻辑
        temp = points[-1]
        points[-1] = points[-3]
        points[-3] = temp
        # 额头中点
        points.append(middleTemple)
        # 额头左半边中间三点
        AddForeheadPointsSub(points, ptMeixin, longAxLeft, shortBy, faceAngle)

    def AddSparsePupilPoints(points):
        LeyeLpts = points[16]
        LeyeRpts = points[20]
        LeyeUpts = points[22]
        LeyeDpts = points[18]

        ReyeLpts = points[28]
        ReyeRpts = points[24]
        ReyeUpts = points[30]
        ReyeDpts = points[26]

        LeyeCpts, ReyeCpts = Point(), Point()
        LeyeCpts.x = (LeyeLpts.x + LeyeRpts.x + LeyeUpts.x + LeyeDpts.x) / 4
        LeyeCpts.y = (LeyeLpts.y + LeyeRpts.y + LeyeUpts.y + LeyeDpts.y) / 4
        ReyeCpts.x = (ReyeLpts.x + ReyeRpts.x + ReyeUpts.x + ReyeDpts.x) / 4
        ReyeCpts.y = (ReyeLpts.y + ReyeRpts.y + ReyeUpts.y + ReyeDpts.y) / 4
        points.append(LeyeCpts)
        points.append(ReyeCpts)

        LeyeCLpts, LeyeCRpts = Point(), Point()
        LeyeCLpts.x = (LeyeCpts.x + LeyeLpts.x) / 2
        LeyeCLpts.y = (LeyeCpts.y + LeyeLpts.y) / 2
        LeyeCRpts.x = (LeyeCpts.x + LeyeRpts.x) / 2
        LeyeCRpts.y = (LeyeCpts.y + LeyeRpts.y) / 2
        points.append(LeyeCLpts)
        points.append(LeyeCRpts)

        ReyeCLpts, ReyeCRpts = Point(), Point()
        ReyeCLpts.x = (ReyeCpts.x + ReyeLpts.x) / 2
        ReyeCLpts.y = (ReyeCpts.y + ReyeLpts.y) / 2
        ReyeCRpts.x = (ReyeCpts.x + ReyeRpts.x) / 2
        ReyeCRpts.y = (ReyeCpts.y + ReyeRpts.y) / 2
        points.append(ReyeCLpts)
        points.append(ReyeCRpts)

    def ConvertPupilSparseToDense(points):
        num = 16

        lcenter = points[215]
        rcenter = points[216]
        lboundary = points[217]
        rboundary = points[219]

        del points[215:221]
        points.append(lcenter)
        points.append(rcenter)

        lradius = math.sqrt(
            pow((lcenter.x - lboundary.x), 2) + pow((lcenter.y - lboundary.y), 2)
        )
        rradius = math.sqrt(
            pow((rcenter.x - rboundary.x), 2) + pow((rcenter.y - rboundary.y), 2)
        )

        lbeta = math.asin((lboundary.y - lcenter.y) / lradius)
        rbeta = math.asin((rboundary.y - rcenter.y) / rradius)

        alpha = 360.0 * M_PI / (num * 180.0)

        for i in range(num):
            lpt = Point()
            angle = lbeta + alpha * i
            lpt.x = lcenter.x - lradius * math.cos(angle)
            lpt.y = lcenter.y + lradius * math.sin(angle)
            points.append(lpt)

        for i in range(num):
            rpt = Point()
            angle = rbeta + alpha * i
            rpt.x = rcenter.x + rradius * math.cos(angle)
            rpt.y = rcenter.y + rradius * math.sin(angle)
            points.append(rpt)

    def AddPupilPoints(points):
        if len(points) <= SPARSE_POINTS_NUM:
            AddSparsePupilPoints(points)
        else:
            ConvertPupilSparseToDense(points)

    points158, points117 = float_2_point(points158), float_2_point(points117)

    merged_points = Merge(points158, points117)
    AddForeheadPoints(merged_points)
    AddPupilPoints(merged_points)

    ret_points = point_2_float(merged_points)

    return ret_points