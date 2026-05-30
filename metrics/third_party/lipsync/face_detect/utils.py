import cv2
import math 
import numpy as np

def draw_text(img, text, size=50, color=(255,255,0), radius=0.5):
    font=cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img,str(text), (size, size), font, radius, color,1)
    return img

def draw_pts_color(img, pts, mode="pts", shift=4, color=(255,255,0), radius=1):
    font=cv2.FONT_HERSHEY_SIMPLEX
    b = 0
    g = 0
    r = 0
    delta = 15 
    phase1 = True
    phase2 = False 
    phase3 = False 
    for cnt,p in enumerate(pts):
        if  phase1 and b + delta < 255: 
            b += delta
        if  phase1 and b + delta >= 255 and g + delta < 255:
            g += delta 
        if phase1 and b + delta >=255 and g + delta>= 255 and r + delta < 255:
            r += delta 

        if phase1 and b + delta >=255 and g + delta>= 255 and r + delta >= 255:
            phase1 = False 
            phase2 = True 
        '''
        if phase2 and b - delta >= 0:
            b -= delta
        if phase2 and b - delta <= 0 and g -delta >=0:
            g -= delta
        if phase2 and b - delta <= 0 and g -delta <=0 and r - delta >=0:
            r -= delta
        '''

        '''
        if phase2 and g - delta >= 0:
            g -= delta
        if phase2 and g - delta <= 0 and b -delta >=0:
            b -= delta
        if phase2 and g - delta <= 0 and b -delta <=0 and r - delta >=0:
            r -= delta
        '''

        #'''
        if phase2 and b - delta >= 0:
            b -= delta
        if phase2 and b - delta <= 0 and r -delta >=0:
            r -= delta
        if phase2 and b - delta <= 0 and r -delta <=0 and g - delta >=0:
            g -= delta
        #'''

        color = (b, g , r)
        cv2.circle(img, (int(p[0] * (1<< shift)),int(p[1] * (1<< shift))) ,radius<<shift, color, 2, cv2.LINE_AA, shift=shift)
    return img

def draw_pts_color2(img, pts, mode=1, shift=4, color=(0,255,255), radius=1):
    b = 0
    g = 0
    r = 0
    delta = 12 
    phase1 = True
    font=cv2.FONT_HERSHEY_SIMPLEX
    for cnt,p in enumerate(pts):
        if mode == 1:
            g += delta
            color = (255, g , 0)
        elif mode == 2:
            r += delta
            color = (255, 255 , r)
        elif mode == 3:
            r += delta
            color = (0, 255 , r)
        elif mode == 4:
            b += delta
            color = (b, 255 , 255)
        else:
            print ('not support { }mode!'.format(mode))
            return
        cv2.circle(img, (int(p[0] * (1<< shift)),int(p[1] * (1<< shift))) ,radius<<shift, color, 2, cv2.LINE_AA, shift=shift)
    return img

def draw_pts(img, pts, mode="pts", shift=4, color=(255,255, 0), radius=1, thickness=1, save_path=None, dif=0):
    for cnt,p in enumerate(pts):
        if mode == "index":
            cv2.circle(img, (int(p[0] * (1 << shift)), int(p[1] * (1 << shift))), radius << shift, color, -1, cv2.LINE_AA, shift=shift)
            cv2.putText(img, str(cnt), (int(float(p[0] + dif)), int(float(p[1] + dif))),cv2.FONT_HERSHEY_SIMPLEX, radius/3, (0,0,255),thickness)
        elif mode == 'pts':
            cv2.circle(img, (int(p[0] * (1<< shift)),int(p[1] * (1<< shift))) ,radius<<shift, color, -1, cv2.LINE_AA, shift=shift)
            #cv2.circle(img, (int(p[0]),int(p[1])) ,1, color,1)
        else:
            print ('not support mode!')
            return
    if(save_path!=None):
        cv2.imwrite(save_path,img)
    return img


def draw_pts_vis(img, pts, vis, mode="pts", shift=4, color=(255,0,0), radius=1, save_path=None, occu_th=0.7):
    for p,v in zip(pts, vis):
        if v < occu_th:
            color = (0,0,255)
        else:
            color = (0,255,0)
        if mode == 'pts':
            cv2.circle(img, (int(p[0] * (1<< shift)),int(p[1] * (1<< shift))) ,int(radius)<<shift, color, -1, cv2.LINE_AA, shift=shift)
        else:
            print ('not support mode!')
            return
    if(save_path!=None):
        cv2.imwrite(save_path,img)
    return img


def draw_rect(img, x1, y1, x2, y2, color=(255,255,0)):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color,1)
    return img

def read_pts(pts_file):
    pts_num = 0
    with open(pts_file, 'r') as pts:
        line = pts.readlines()[1]
        pts_num = int(line[:-1].split(":")[1])
    with open(pts_file, 'r') as pts:
        lines = pts.readlines()[3:pts_num+3]
        pt = []
        for line in lines:
            pt.append(line.strip('\n').split(' '))
        pt = np.array(pt, dtype='float32')
    return pt

def write_pts(pts_file, out_pt):
    with open(pts_file, 'w', encoding='utf-8') as out_pts:
        out_pts.write('version: 1\n')
        out_pts.write('n_points: ' + str(len(out_pt)) + '\n')
        out_pts.write('{\n')
        for line in out_pt:
            out_pts.write('%f %f\n' % (line[0], line[1]))
        out_pts.write('}\n')

def dis_pts(pt1, pt2):
    x1 = float(pt1[0]) 
    y1 = float(pt1[1])
    x2 = float(pt2[0])
    y2 = float(pt2[1])
    dis = math.sqrt(pow((x1-x2), 2) + pow((y1-y2), 2))
    return dis

def get_pts5(pts):
    if len(pts) == 5:
        fa5p = pts
    elif len(pts) == 90 or len(pts) == 94:
        fa5p = np.array([
            pts[16] * 0.5 + pts[20] * 0.5,
            pts[24] * 0.5 + pts[28] * 0.5,
            pts[32],
            pts[45],
            pts[51]], dtype=np.float32)
    elif len(pts) == 256:
        fa5p = np.array([
            pts[32] * 0.5 + pts[44] * 0.5,
            pts[56] * 0.5 + pts[68] * 0.5,
            pts[80],
            pts[102],
            pts[120]], dtype=np.float32)
    else:
        raise ValueError("[Error]Invalid Pts(%d)!" % len(pts))
    return fa5p



###########################   N点对齐 ########################
#求矩阵的逆
def inv_affine(M):
    #添加一行
    M = np.r_[M,np.array([[0,0,1.0]])]
    return np.linalg.inv(M)

def landmark_warpAffine(land,M):
    new_n = []
    for i in range(len(land)):
        pts = []    
        pts.append(np.squeeze(np.array(M[0]))[0]*land[i][0]+np.squeeze(np.array(M[0]))[1]*land[i][1]+np.squeeze(np.array(M[0]))[2])
        pts.append(np.squeeze(np.array(M[1]))[0]*land[i][0]+np.squeeze(np.array(M[1]))[1]*land[i][1]+np.squeeze(np.array(M[1]))[2])
        new_n.append(pts)
    return np.array(new_n)

def transformation_from_points(points1, points2):
    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)
    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2
    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2
    #points2 = np.array(points2)
    #write_pts('pt2.txt', points2)
    U, S, Vt = np.linalg.svd(points1.T * points2)
    R = (U * Vt).T
    return np.vstack([np.hstack(((s2 / s1) * R,c2.T - (s2 / s1) * R * c1.T)),np.matrix([0., 0., 1.])])


def align_N(img_im, orgi_landmarks, tar_landmarks, align_h, align_w, rnd_scale=1):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))

    #random scale
    #rnd_scale = 1 #1.2 #random.uniform(self.scale[0], self.scale[1])
    dx = (rnd_scale * align_w - align_w) / 2.0
    dy = (rnd_scale * align_h - align_h) / 2.0
    tar_landmarks_scale = tar_landmarks.copy() 
    tar_landmarks_scale += [dx, dy]
    tar_landmarks_scale /= [rnd_scale, rnd_scale]
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks_scale]))
    M = transformation_from_points(pts1, pts2)
    M = M[:2] 

    dst = cv2.warpAffine(img_im, M, (align_h, align_w))
    #dst = cv2.warpAffine(img_im, M, (int(img_im.shape[0]), int(img_im.shape[1])))

    #warp pts
    #dst_pts = landmark_warpAffine(orgi_landmarks, M)

    return dst, M # dst_pts


def align_N_aug(img_im, orgi_landmarks, tar_landmarks, align_size, rnd_scale=1.0):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    M = M[:2] 
    pts_phase1 = landmark_warpAffine(orgi_landmarks, M)
    
    crop_face, crop_M = crop_part(img_im, pts_phase1, M, align_size, rnd_scale)
    
    return crop_face, crop_M


def align_N_picasso(img_im, orgi_landmarks, tar_landmarks, align_size, rnd_scale=1.0):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    M = M[:2] 
    pts_phase1 = landmark_warpAffine(orgi_landmarks, M)
    
    left_part, left_M = crop_part(img_im, np.concatenate((pts_phase1[0:8,:], pts_phase1[16:24,:]),axis=0), M, align_size, rnd_scale)
    right_part, right_M = crop_part(img_im, np.concatenate((pts_phase1[8:16,:], pts_phase1[24:32,:]),axis=0), M, align_size, rnd_scale, flip=True)
    mouth, mouth_M = crop_part(img_im, pts_phase1[54:76], M, align_size, rnd_scale)    
    
    return [left_part, right_part, mouth], [left_M, right_M, mouth_M]


def align_N_picasso_aug2(img_im, orgi_landmarks, tar_landmarks, align_size, rnd_scale=1.0):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    M = M[:2] 
    pts_phase1 = landmark_warpAffine(orgi_landmarks, M)
    
    left_part, left_M = crop_part_aug2(img_im, np.concatenate((pts_phase1[0:8,:], pts_phase1[16:24,:]),axis=0), M, align_size, rnd_scale)
    right_part, right_M = crop_part_aug2(img_im, np.concatenate((pts_phase1[8:16,:], pts_phase1[24:32,:]),axis=0), M, align_size, rnd_scale, flip=True)
    mouth, mouth_M = crop_part_aug2(img_im, pts_phase1[54:76], M, align_size, rnd_scale)    
    
    return [left_part, right_part, mouth], [left_M, right_M, mouth_M]


def align_N_picasso_aug3(img_im, orgi_landmarks, tar_landmarks, align_size, rnd_scale=1.0):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    M = M[:2] 
    pts_phase1 = landmark_warpAffine(orgi_landmarks, M)
    
    left_part, left_M, _ = crop_part_aug3(img_im, np.concatenate((pts_phase1[0:8,:], pts_phase1[16:24,:]),axis=0), M, align_size, rnd_scale)
    right_part, right_M, _ = crop_part_aug3(img_im, np.concatenate((pts_phase1[8:16,:], pts_phase1[24:32,:]),axis=0), M, align_size, rnd_scale, flip=True)
    mouth, mouth_M, mouth_info = crop_part_aug3(img_im, pts_phase1[54:76], M, align_size, rnd_scale)    
    
    return [left_part, right_part, mouth], [left_M, right_M, mouth_M], mouth_info


def align_N_picasso_aug4(img_im, orgi_landmarks, tar_landmarks, net_size, rnd_scale=1.0):
    data_dir = "data/align_all_data_0925/"
    meanf = data_dir + "/mean_pts130_scale112_full_flip_phase2.txt" 

    align_size = [112, 112]
    mean_landmarks_eye = read_pts(meanf.replace("phase2", "eye_brow")) * [align_size[0]/112.0, align_size[1]/112.0]
    mean_landmarks_reye = read_pts(meanf.replace("phase2", "reye_brow")) * [align_size[0]/112.0, align_size[1]/112.0]
    mean_landmarks_mouth = read_pts(meanf.replace("phase2", "mouth")) * [align_size[0]/112.0, align_size[1]/112.0]

    #leye
    orgi_eye_landmarks = np.concatenate((orgi_landmarks[0:8,:], orgi_landmarks[16:24,:]),axis=0)
    tar_eye_landmarks = mean_landmarks_eye 
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_eye_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_eye_landmarks]))
    M_eye = transformation_from_points(pts1, pts2)
    M_eye = M_eye[:2] 
    tar_eye_landmarks = landmark_warpAffine(orgi_eye_landmarks, M_eye)

     #reye
    orgi_reye_landmarks = np.concatenate((orgi_landmarks[8:16,:], orgi_landmarks[24:32,:]),axis=0)
    tar_reye_landmarks = mean_landmarks_reye 
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_reye_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_reye_landmarks]))
    M_reye = transformation_from_points(pts1, pts2)
    M_reye = M_reye[:2] 
    tar_reye_landmarks = landmark_warpAffine(orgi_reye_landmarks, M_reye)
   
    #mouth
    orgi_mouth_landmarks = orgi_landmarks[54:76] 
    tar_mouth_landmarks = mean_landmarks_mouth 
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_mouth_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_mouth_landmarks]))
    M_mouth = transformation_from_points(pts1, pts2)
    M_mouth = M_mouth[:2] 
    tar_mouth_landmarks = landmark_warpAffine(orgi_mouth_landmarks, M_mouth)

    #crop mouth/eye and aug
    left_part, left_M = crop_part_aug2(img_im, tar_eye_landmarks, M_eye, net_size, rnd_scale)
    right_part, right_M = crop_part_aug2(img_im, tar_reye_landmarks, M_reye, net_size, rnd_scale, flip=True)
    mouth, mouth_M = crop_part_aug2(img_im, tar_mouth_landmarks, M_mouth, net_size, rnd_scale)

    return [left_part, right_part, mouth], [left_M, right_M, mouth_M]


def align_N_picasso_aug5(img_im, orgi_landmarks, tar_landmarks, align_size, rnd_scale=1.0):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    M = M[:2] 
    pts_phase1 = landmark_warpAffine(orgi_landmarks, M)
    
    left_part, left_M = crop_part_aug5(img_im, np.concatenate((pts_phase1[0:8,:], pts_phase1[16:24,:]),axis=0), M, align_size, rnd_scale)
    right_part, right_M = crop_part_aug5(img_im, np.concatenate((pts_phase1[8:16,:], pts_phase1[24:32,:]),axis=0), M, align_size, rnd_scale, flip=True)
    mouth, mouth_M = crop_part_aug5(img_im, pts_phase1[54:76], M, align_size, rnd_scale)    
    
    return [left_part, right_part, mouth], [left_M, right_M, mouth_M]



def align_N_picassoV2(img_im,  orgi_landmarks, tar_landmarks, align_size, pred_p1, rnd_scale=1.0):
    pts1 = np.float64(np.matrix([[point[0], point[1]] for point in orgi_landmarks]))
    pts2 = np.float64(np.matrix([[point[0], point[1]] for point in tar_landmarks]))
    M = transformation_from_points(pts1, pts2)
    M = M[:2] 
    pts_phase1 = landmark_warpAffine(orgi_landmarks, M)
    
    left_part, left_M = crop_part(img_im, np.concatenate((pts_phase1[0:8,:], pts_phase1[16:24,:]),axis=0), M, align_size, rnd_scale)
    right_part, right_M = crop_part(img_im, np.concatenate((pts_phase1[8:16,:], pts_phase1[24:32,:]),axis=0), M, align_size, rnd_scale, flip=True)
    mouth, mouth_M = crop_part(img_im, pts_phase1[54:76], M, align_size, rnd_scale)    

    #pred p1 global info
    left_p1 = np.concatenate((pred_p1[0:8,:], pred_p1[16:24,:]), axis=0)
    right_p1 = np.concatenate((pred_p1[8:16,:], pred_p1[24:32,:]), axis=0)
    mouth_p1 = pred_p1[54:76]
    left_p1 = landmark_warpAffine(left_p1, left_M)
    right_p1 = landmark_warpAffine(right_p1, right_M)
    mouth_p1 = landmark_warpAffine(mouth_p1, mouth_M)
    right_p1[:,0] = right_part.shape[1] - 1 - right_p1[:,0]

    #gen heatmap
    left_heatmap = gen_heatmap(left_part, left_p1)
    right_heatmap = gen_heatmap(right_part, right_p1)
    mouth_heatmap = gen_heatmap(mouth, mouth_p1)

    return [left_part, right_part, mouth], [left_M, right_M, mouth_M]


def gen_heatmap(img, pts):
    h, w = img.shape
    heatmap = np.zeros(shape=(h, w))
    for pt in pts:
        px = int(pt[0])
        py = int(pt[1])
        if (px > 0 and px < w) and (py > 0 and py < h): 
            heatmap[py, px] = 255
    return heatmap 

def crop_part(img, pts, p1_M, align_size, rnd_scale, flip=False):
    xmin = min(pts[:,0])
    xmax = max(pts[:,0])
    ymin = min(pts[:,1])
    ymax = max(pts[:,1])

    
    width = max(xmax - xmin, ymax - ymin) * rnd_scale 
    width = 2 if width == 0 else width 
    scale = align_size / width

    #random shift:
    cx = (xmax + xmin) / 2.0
    cy = (ymax + ymin) / 2.0
    xOffset = -(cx - width / 2.0)
    yOffset = -(cy - width / 2.0)
    
    M = p1_M.copy()
    M[0,0]= M[0,0] * scale
    M[0,1]= M[0,1] * scale
    M[0,2]= M[0,2] * scale + xOffset *scale
    M[1,0]= M[1,0] * scale
    M[1,1]= M[1,1] * scale
    M[1,2]= M[1,2] * scale + yOffset *scale

    if flip:
        M[0,0] = -M[0,0]
        M[0,1] = -M[0,1]
        M[0,2] = -M[0,2] + align_size-1
    dst = cv2.warpAffine(img, M, (align_size, align_size))
    return dst,  M 


def crop_part_aug2(img, pts, p1_M, align_size, rnd_scale, flip=False):
    xmin = min(pts[:,0])
    xmax = max(pts[:,0])
    ymin = min(pts[:,1])
    ymax = max(pts[:,1])

    width = (xmax - xmin) * rnd_scale 
    height = (ymax - ymin) * rnd_scale 
    width = 6 if width < 6 else width 
    height = 6 if height < 6 else height 

    scale_width = align_size / (width * 1.0)
    scale_height = align_size / (height * 1.0)

    #random shift:
    cx = (xmax + xmin) / 2.0
    cy = (ymax + ymin) / 2.0
    xOffset = -(cx - width / 2.0)
    yOffset = -(cy - height / 2.0)

    #M = M1 * M2: 将对齐和crop_part合二为一
    M = p1_M.copy()
    M[0,0]= M[0,0] * scale_width
    M[0,1]= M[0,1] * scale_width
    M[0,2]= M[0,2] * scale_width + xOffset * scale_width
    M[1,0]= M[1,0] * scale_height
    M[1,1]= M[1,1] * scale_height
    M[1,2]= M[1,2] * scale_height + yOffset * scale_height

    if flip:
        M[0,0] = -M[0,0]
        M[0,1] = -M[0,1]
        M[0,2] = -M[0,2] + align_size - 1
    dst = cv2.warpAffine(img, M, (align_size, align_size))
    return dst,  M 


def crop_part_aug3(img, pts, p1_M, align_size, rnd_scale, flip=False):
    xmin = min(pts[:,0])
    xmax = max(pts[:,0])
    ymin = min(pts[:,1])
    ymax = max(pts[:,1])

    width = (xmax - xmin) * rnd_scale 
    height = (ymax - ymin) * rnd_scale 
    width = 6 if width < 6 else width 
    height = 6 if height < 6 else height 

    info = "no 1/2"
    if height < width * 0.5:
        height = width * 0.5
        info = "height/2 " +  str(len(pts))
    if width < height * 0.5:
        width = height * 0.5
        info = "height/2 " +  str(len(pts))

    scale_width = align_size / (width * 1.0)
    scale_height = align_size / (height * 1.0)

    #random shift:
    cx = (xmax + xmin) / 2.0
    cy = (ymax + ymin) / 2.0
    xOffset = -(cx - width / 2.0)
    yOffset = -(cy - height / 2.0)

    #M = M1 * M2: 将对齐和crop_part合二为一
    M = p1_M.copy()
    M[0,0]= M[0,0] * scale_width
    M[0,1]= M[0,1] * scale_width
    M[0,2]= M[0,2] * scale_width + xOffset * scale_width
    M[1,0]= M[1,0] * scale_height
    M[1,1]= M[1,1] * scale_height
    M[1,2]= M[1,2] * scale_height + yOffset * scale_height

    if flip:
        M[0,0] = -M[0,0]
        M[0,1] = -M[0,1]
        M[0,2] = -M[0,2] + align_size - 1
    dst = cv2.warpAffine(img, M, (align_size, align_size))
    return dst,  M, info 


def crop_part_aug5(img, pts, p1_M, align_size, rnd_scale, flip=False):
    xmin = min(pts[:,0])
    xmax = max(pts[:,0])
    ymin = min(pts[:,1])
    ymax = max(pts[:,1])

    width = (xmax - xmin) * rnd_scale 
    height = (ymax - ymin) * (rnd_scale + 0.2) 
    width = 6 if width < 6 else width 
    height = 6 if height < 6 else height 

    scale_width = align_size / (width * 1.0)
    scale_height = align_size / (height * 1.0)

    #random shift:
    cx = (xmax + xmin) / 2.0
    cy = (ymax + ymin) / 2.0
    xOffset = -(cx - width / 2.0 + 0.25)
    yOffset = -(cy - height / 2.0 + 0.25)

    #M = M1 * M2: 将对齐和crop_part合二为一
    M = p1_M.copy()
    M[0,0]= M[0,0] * scale_width
    M[0,1]= M[0,1] * scale_width
    M[0,2]= M[0,2] * scale_width + xOffset * scale_width
    M[1,0]= M[1,0] * scale_height
    M[1,1]= M[1,1] * scale_height
    M[1,2]= M[1,2] * scale_height + yOffset * scale_height

    if flip:
        M[0,0] = -M[0,0]
        M[0,1] = -M[0,1]
        M[0,2] = -M[0,2] + align_size - 1
    dst = cv2.warpAffine(img, M, (align_size, align_size))
    return dst,  M 

###########################   N点对齐 ########################

def enlarge_bbox(x1, y1, x2, y2, imgh, imgw, ratio=1.2):
    w = abs(x2 - x1)
    h = abs(y2 - y1)

    dw = (ratio - 1.0)  * w
    x1 -= dw/2.0
    x2 += dw/2.0
    x1 = int(max(0, x1))
    x2 = int(min(imgw - 1, x2))

    dh = (ratio - 1.0) * h 
    y1 -= dh/2.0
    y2 += dh/2.0
    y1 = int(max(0, y1))
    y2 = int(min(imgh - 1, y2))
    return x1, y1, x2, y2


def warp_by_rect(img, x1, y1, x2, y2, netWidth, enlarge=1.3):
    xmin = x1 
    xmax = x2 
    ymin = y1 
    ymax = y2 

    #drop forehead
    ymin = y1 + (y2 - y1) * 0.3 #0.15

    width = max(xmax - xmin, ymax - ymin) * enlarge 
    width = 2 if width == 0 else width
    scale = netWidth / width 

    #print (xmin, xmax, ymin, ymax,  "+++")
    cx = (xmax + xmin) / 2.0
    cy = (ymax + ymin) / 2.0
    xOffset = -(cx - width / 2.0)
    yOffset = -(cy - width / 2.0)

    M = np.zeros((2,3))
    M[0,0]= scale
    M[0,1]= 0 
    M[0,2]= xOffset * scale
    M[1,0]= 0 
    M[1,1]= scale 
    M[1,2]= yOffset * scale

    dst = cv2.warpAffine(img, M, (netWidth, netWidth))
    #dst1 = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    #cv2.imwrite('face1.jpg', dst1)
    #print (cx, cy, width, xOffset, yOffset, scale, "+++")
    return dst,  M


def cvt256PtsTo94Pts(facePoints256):
    facePoints94 = []
    if 0 == len(facePoints256):
        return 0

    pts94NumPtNose = 13
    noseIndex = [0, 4, 18, 19, 7, 8, 10, 11, 12, 14, 15, 21, 20]

    pts256NumPtEyeBrow = 16
    pts256NumPtEye = 24
    pts256NumPtNose = 22
    pts256NumPtMouth = 72
    pts256NumPtProfile = 41
    pts256NumPtPupil = 34  # --> 228:6; 256:34
    pts256NumPtForehead = 7

    row = 0

    # for (int i = 0 i < pts256NumPtEyeBrow * 2 i++, row++)
    for i in range(0, pts256NumPtEyeBrow * 2):
        if (i % 2):
            row += 1
            continue
        facePoints94.append(facePoints256[row])
        row += 1

    for i in range(pts256NumPtEye * 2):  # i++, row++)
        row += 1
        if (i % 3):
            continue
        facePoints94.append(facePoints256[row])

    # std::vector<cv::Point2f> nosePts
    # nosePts.clear()
    # nosePts.reserve(pts256NumPtNose)
    nosePts = []
    # for (int i = 0 i < pts256NumPtNose i++, row++)
    for i in range(0, pts256NumPtNose):  # i++, row++)
        nosePts.append(facePoints256[row])
        row += 1

    # nose fix: 94点和130点的鼻翼点略有不同，作以下修正，不影响外围逻辑
    nosePts[8][0] = (nosePts[8][0] + nosePts[9][0]) / 2
    nosePts[8][1] = (nosePts[8][1] + nosePts[9][1]) / 2
    nosePts[14][0] = (nosePts[14][0] + nosePts[13][0]) / 2
    nosePts[14][1] = (nosePts[14][1] + nosePts[13][1]) / 2

    # for (int j =0 j<pts94NumPtNosej++)
    for j in range(0, pts94NumPtNose):  # j++)
        facePoints94.append(nosePts[noseIndex[j]])

    # Mouth
    # for (int i = 0 i < pts256NumPtMouth i++, row++)
    for i in range(0, pts256NumPtMouth):  # i++, row++)
        if (i % 3 or 36 == i or 54 == i):
            row += 1
            continue
        facePoints94.append(facePoints256[row])
        row += 1

    # Profile
    # for (int i = 0 i < pts256NumPtProfile i++, row++)
    for i in range(0, pts256NumPtProfile):  # i++, row++)
        if (i % 2):
            row += 1
            continue
        facePoints94.append(facePoints256[row])
        row += 1

    # forehead
    # std::vector<cv::Point2f> foreheadPts
    # foreheadPts.clear()
    # foreheadPts.reserve(pts256NumPtForehead)
    foreheadPts = []
    # for (int i = 0 i < pts256NumPtForehead i++, row++)
    for i in range(0, pts256NumPtForehead):  # i++, row++)
        foreheadPts.append(facePoints256[row])
        row += 1

    # //pupil
    # for (int i = 0 i < pts256NumPtPupil i++, row++)
    for i in range(0, pts256NumPtPupil):  # i++, row++)
        if (i < 3 or 9 == i or 18 == i or 25 == i):  # --> 228:close; 256:open
            facePoints94.append(facePoints256[row])
        # facePoints94.append(facePoints256[row])
        row += 1

    facePoints94 = np.array(facePoints94)
    facePoints94 = facePoints94.reshape(94, 2)

    return facePoints94

def cvt221PtsTo94Pts(facePoints221):
    facePoints94 = []
    if 0 == len(facePoints221):
        return 0

    pts94NumPtNose = 13
    noseIndex = [0, 4, 18, 19, 7, 8, 10, 11, 12, 14, 15, 21, 20]

    pts256NumPtEyeBrow = 16
    pts256NumPtEye = 24
    pts256NumPtNose = 22
    pts256NumPtMouth = 72
    pts256NumPtProfile = 41
    pts256NumPtPupil = 6  # 34 --> 228:6; 256:34
    pts256NumPtForehead = 7

    row = 0

    # for (int i = 0 i < pts256NumPtEyeBrow * 2 i++, row++)
    for i in range(0, pts256NumPtEyeBrow * 2):
        if (i % 2):
            row += 1
            continue
        facePoints94.append(facePoints221[row])
        row += 1

    for i in range(pts256NumPtEye * 2):  # i++, row++)
        row += 1
        if (i % 3):
            continue
        facePoints94.append(facePoints221[row])

    # std::vector<cv::Point2f> nosePts
    # nosePts.clear()
    # nosePts.reserve(pts256NumPtNose)
    nosePts = []
    # for (int i = 0 i < pts256NumPtNose i++, row++)
    for i in range(0, pts256NumPtNose):  # i++, row++)
        nosePts.append(facePoints221[row])
        row += 1

    # nose fix: 94点和130点的鼻翼点略有不同，作以下修正，不影响外围逻辑
    nosePts[8][0] = (nosePts[8][0] + nosePts[9][0]) / 2
    nosePts[8][1] = (nosePts[8][1] + nosePts[9][1]) / 2
    nosePts[14][0] = (nosePts[14][0] + nosePts[13][0]) / 2
    nosePts[14][1] = (nosePts[14][1] + nosePts[13][1]) / 2

    # for (int j =0 j<pts94NumPtNosej++)
    for j in range(0, pts94NumPtNose):  # j++)
        facePoints94.append(nosePts[noseIndex[j]])

    # Mouth
    # for (int i = 0 i < pts256NumPtMouth i++, row++)
    for i in range(0, pts256NumPtMouth):  # i++, row++)
        if (i % 3 or 36 == i or 54 == i):
            row += 1
            continue
        facePoints94.append(facePoints221[row])
        row += 1

    # Profile
    # for (int i = 0 i < pts256NumPtProfile i++, row++)
    for i in range(0, pts256NumPtProfile):  # i++, row++)
        if (i % 2):
            row += 1
            continue
        facePoints94.append(facePoints221[row])
        row += 1

    # forehead
    # std::vector<cv::Point2f> foreheadPts
    # foreheadPts.clear()
    # foreheadPts.reserve(pts256NumPtForehead)
    # foreheadPts = []
    # # for (int i = 0 i < pts256NumPtForehead i++, row++)
    # for i in range(0, pts256NumPtForehead):  # i++, row++)
    #     foreheadPts.append(facePoints256[row])
    #     row += 1

    # //pupil
    # for (int i = 0 i < pts256NumPtPupil i++, row++)
    for i in range(0, pts256NumPtPupil):  # i++, row++)
        # if (i < 3 or 9 == i or 18 == i or 25 == i): #--> 228:close; 256:open
        # facePoints94.append(facePoints256[row])
        facePoints94.append(facePoints221[row])
        row += 1

    facePoints94 = np.array(facePoints94)
    facePoints94 = facePoints94.reshape(94, 2)

    return facePoints94


def cvt130PtsTo94Pts(facePoints130):
    facePoints94 = []

    if (0 == len(facePoints130)):
        return 0

    pts94NumPtNose = 13
    noseIndex = [0, 4, 18, 19, 7, 8, 10, 11, 12, 14, 15, 21, 20]

    pts130NumPtEyeBrow = 8
    pts130NumPtEye = 8
    pts130NumPtNose = 22
    pts130NumPtMouth = 22
    pts130NumPtProfile = 41
    pts130NumPtPupil = 6
    pts130NumPtForehead = 7

    row = 0
    # EyeBrow
    # for (i = 0 i < pts130NumPtEyeBrow * 2 i++, row++)
    for i in range(0, pts130NumPtEyeBrow * 2):
        facePoints94.append(facePoints130[row])
        row += 1

    # Eye
    # for (i = 0 i < pts130NumPtEye * 2 i++, row++)
    for i in range(0, pts130NumPtEye * 2):
        facePoints94.append(facePoints130[row])
        row += 1

    # nose
    nosePts = []
    # for (i = 0 i < pts130NumPtNose i++, row++)
    for i in range(0, pts130NumPtNose):
        nosePts.append(facePoints130[row])
        row += 1

    # nose fix: 94点和130点的鼻翼点略有不同，作以下修正，不影响外围逻辑
    nosePts[8][0] = (nosePts[8][0] + nosePts[9][0]) / 2
    nosePts[8][1] = (nosePts[8][1] + nosePts[9][1]) / 2
    nosePts[14][0] = (nosePts[14][0] + nosePts[13][0]) / 2
    nosePts[14][1] = (nosePts[14][1] + nosePts[13][1]) / 2

    # for (i = 0 i < pts94NumPtNose i++)
    for i in range(0, pts94NumPtNose):
        facePoints94.append(nosePts[noseIndex[i]])

    # mouth
    # for (i = 0 i < pts130NumPtMouth i++, row++)
    for i in range(0, pts130NumPtMouth):
        facePoints94.append(facePoints130[row])
        row += 1

    # profile
    # for (i = 0 i < pts130NumPtProfile i++, row++)
    for i in range(0, pts130NumPtProfile):
        if (i % 2):
            row += 1
            continue
        facePoints94.append(facePoints130[row])
        row += 1

    # pupil
    facePoints94.append((facePoints94[16] + facePoints94[20]) / 2)
    facePoints94.append((facePoints94[24] + facePoints94[28]) / 2)

    facePoints94.append((facePoints94[16] + facePoints94[88]) / 2)
    facePoints94.append((facePoints94[20] + facePoints94[88]) / 2)

    facePoints94.append((facePoints94[24] + facePoints94[89]) / 2)
    facePoints94.append((facePoints94[28] + facePoints94[89]) / 2)

    facePoints94 = np.array(facePoints94)
    facePoints94 = facePoints94.reshape(94, 2)

    return facePoints94