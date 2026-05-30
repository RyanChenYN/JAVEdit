import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import sys
import torch
import numpy as np
# import load_model
from .utils import read_pts, cvt256PtsTo94Pts, cvt130PtsTo94Pts, align_N, align_N_aug, align_N_picasso_aug3, landmark_warpAffine, inv_affine
from .align_tools import points_117_158_256
# from .scrfd import SCRFDONNX
from .yoloface import YoloFace
import cv2
import time
from .model_landmark import Model as LandmarkModel

def phase1(device, p1_path):
    align_w = 256
    align_h = 256
    net_scale = 1.1
    in_channel = 3
    meanf = os.path.join(BASE_DIR, 'meanfiles/mean_pts130_scale112_full_flip_phase1.txt')
    # model_path = os.path.join(BASE_DIR, 'weights/p1.pt')
    # model_path = os.path.join(BASE_DIR, 'models/1202_dzg/p1.pkl')
    mean_ld = read_pts(meanf) * [align_h/112.0, align_w/112.0]
    model = torch.jit.load(p1_path)
    model.to(device)
    model.eval().half()
    return model, mean_ld, align_w, align_h, net_scale, in_channel

def phase2(device, p2_path):
    align_w = 256
    align_h = 256
    net_scale = 1.5
    in_channel = 9
    meanf = os.path.join(BASE_DIR, 'meanfiles/mean_pts130_scale112_full_flip_phase2.txt')
    # model_path = os.path.join(BASE_DIR, 'weights/p2.pt')
    # model_path = os.path.join(BASE_DIR, 'models/1202_dzg/p2.pkl')
    mean_ld = read_pts(meanf) * [align_h/112.0, align_w/112.0]

    '''
    img = np.zeros((256, 256, 3))
    utils.draw_pts(img, mean_ld)
    cv2.imwrite('mean.jpg', img)
    '''

    model = torch.jit.load(p2_path)
    model.to(device)
    model.eval().half()
    return model, mean_ld, align_w, align_h, net_scale, in_channel


def cvt221PtsTo130Pts(pts221):
    pts130 = []
    j = -1
    #eyebrow
    for i in range(0, 16 * 2):
        j += 1
        if (i % 2):
            continue
        pts130.append(pts221[j])
    #eye
    for i in range(0, 24 * 2):
        j += 1
        if (i % 3):
            continue
        pts130.append(pts221[j])
    #nose
    for i in range(0, 22):
        j += 1
        pts130.append(pts221[j])
    #mouth
    for i in range(0, 72):
        j += 1
        if (i % 3 or i == 36 or i == 54):
            continue
        pts130.append(pts221[j])
    #profile
    for i in range(0, 41):
        j += 1
        pts130.append(pts221[j])

    #forehead
    for i in range(0, 7):
        pts130.append(np.array([0,0]))

    #pupil
    for i in range(0, 6):
        pts130.append(np.array([0,0]))

    pts130 = np.array(pts130)
    return pts130


def cvt221PtsTo228Pts(pts221):
    pts228 = []
    j = -1
    #eye
    for i in range(0, 40 * 2):
        j += 1
        pts228.append(pts221[j])
    #nose
    for i in range(0, 22):
        j += 1
        pts228.append(pts221[j])

    #mouth
    for i in range(0, 72):
        j += 1
        pts228.append(pts221[j])

    #profile
    for i in range(0, 41):
        j += 1
        pts228.append(pts221[j])

    #forehead
    for i in range(0, 7):
        pts228.append(np.array([0,0]))

    #pupil
    for i in range(0, 6):
        j += 1
        pts228.append(pts221[j])
    pts228 = np.array(pts228)
    return pts228


def cvt_pts(pts221):
    pred_p2_eye = pts221[0:80,:]
    pred_p1_nose = pts221[80:102,:]
    pred_p2_mouth = pts221[102:174,:]
    pred_p1_profile = pts221[174:215,:]
    pred_p2_pupil = pts221[215:221,:]

    pred_p2 = np.concatenate((pred_p2_eye, pred_p2_mouth, pred_p2_pupil))

    pts221 = np.concatenate((pred_p2[0:16, :], pred_p2[43:59, :], pred_p2[16:40, :],pred_p2[59:83, :], pred_p1_nose, pred_p2[86:158,:], pred_p1_profile, pred_p2[40:41,:], pred_p2[83:84,:],pred_p2[41:43,:], pred_p2[84:86,:]), axis=0)

    # pts130 = cvt221PtsTo130Pts(pts221)
    # pts228 = cvt221PtsTo228Pts(pts221)
    # pts = np.concatenate((pts228, pts130))
    return pts221

class RefinePts(object):
    def __init__(self, device='cuda', p1_path ='checkpoints/p1.pt', p2_path='checkpoints/p2.pt'):
        self.test_device = torch.device(device if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available() and 'cuda' in device:
            torch.backends.cudnn.benchmark = True

        self.model1, self.mean_ld1, self.net1_w, self.net1_h, self.net1_scale, self.net1_c = phase1(self.test_device, p1_path)
        self.model2, self.mean_ld2, self.net2_w, self.net2_h, self.net2_scale, self.net2_c = phase2(self.test_device, p2_path)
        self.mean_ld0 = read_pts(os.path.join(BASE_DIR, 'meanfiles/face_mean_5.txt')) * [self.net1_w / 112.0, self.net1_h / 112.0]

    # return 221 points
    @torch.no_grad()
    def __call__(self, im_cv2, pts_list):
        pts_res_list = []
        pts_score_list = []
        for pts in pts_list:
            pre_face = False
            pre_pts = None
            dup = 0
            pre_conf1 = 0
            pre_conf2 = 0
            while (dup < 3):
                dup += 1
                cur_pts_p1 = []
                cur_vis_p1 = []
                cur_pts_p2 = []
                cur_vis_p2 = []

                img = im_cv2
                if pre_pts is None:

                    pre_pts = pts
                    face, M = align_N(img, pre_pts, self.mean_ld0, self.net1_h, self.net1_w, rnd_scale=self.net1_scale)
                    # cv2.imwrite('show_dir/{}_face.jpg'.format(cnt), face)
                else:
                    pre_pts = pre_pts[:117]
                    face, M = align_N_aug(img, pre_pts, self.mean_ld1, self.net1_h, rnd_scale=self.net1_scale)

                # phase1
                face = np.reshape(face, (face.shape[0], face.shape[1], self.net1_c))
                face = np.float32(face)
                x = face / 128.0 - 1.0
                x = x.transpose((2, 0, 1))
                x = np.expand_dims(x, axis=0)
                x = torch.from_numpy(x)
                x = x.float().to(self.test_device)

                pts_phase1, pred_label, vis_phase1 = self.model1(x.half())
                label = torch.sigmoid(pred_label)
                label = label.float().cpu().numpy()[0][0]

                res = pts_phase1.float().cpu().numpy()[0]
                for i in range(len(res) // 2):
                    origin = np.float32([res[2 * i], res[2 * i + 1]])
                    cur_pts_p1.append(origin)
                cur_pts_p1 = np.asarray(cur_pts_p1)
                # cur_vis_p1 = torch.sigmoid(vis_phase1).cpu().numpy()[0]
                M_ = inv_affine(M)
                cur_pts_p1 = landmark_warpAffine(cur_pts_p1, M_)

                # phase2
                parts, M, _ = align_N_picasso_aug3(img, cur_pts_p1[:76], self.mean_ld2, self.net2_h, rnd_scale=self.net2_scale)
                for i in range(3):
                    parts[i] = parts[i][:, :, np.newaxis]
                face2 = np.concatenate((parts[0], parts[1], parts[2]), axis=-1)
                face2 = np.reshape(face2, (face2.shape[0], face2.shape[1], self.net2_c))
                face2 = np.float32(face2)
                x = face2 / 128.0 - 1.0
                x = x.transpose((2, 0, 1))
                x = np.expand_dims(x, axis=0)
                x = torch.from_numpy(x)
                x = x.float().to(self.test_device)
                pts_phase2, vis_phase2 = self.model2(x.half())
                res = pts_phase2.float().cpu().numpy()[0]
                for i in range(len(res) // 2):
                    origin = np.float32([res[2 * i], res[2 * i + 1]])
                    cur_pts_p2.append(origin)
                cur_pts_p2 = np.asarray(cur_pts_p2)
                # cur_vis_p2 = torch.sigmoid(vis_phase2).cpu().numpy()[0]
                M = np.array(M)
                M_ = M.copy()
                for i in range(3):
                    M_[i] = inv_affine(M[i])[:2]
                cur_pts_p2[0:43] = landmark_warpAffine(cur_pts_p2[0:43], M_[0])
                cur_pts_p2[43:86] = landmark_warpAffine(cur_pts_p2[43:86], M_[1])
                cur_pts_p2[86:158] = landmark_warpAffine(cur_pts_p2[86:158], M_[2])

                # update
                pre_pts = cur_pts_p1

                pre_face = True
                if(abs(label-pre_conf1)<0.0001 and abs(pre_conf2-pre_conf1)<0.0001 and label>0.85):
                    break
                pre_conf2 = pre_conf1
                pre_conf1 = label

            if pre_face:
                pts_score_list.append(pre_conf1)
                # cur_pts = np.concatenate((cur_pts_p2[0:80, :], cur_pts_p1[32:54, :], cur_pts_p2[80:152, :], cur_pts_p1[76:117, :], cur_pts_p2[152:158, :]), axis=0)
                # #
                # # cur_vis = np.concatenate(
                # #     (cur_vis_p2[0:80], cur_vis_p1[32:54], cur_vis_p2[80:152], cur_vis_p1[76:117], cur_vis_p2[152:158]),
                # #     axis=0)
                # #
                # # cur_vis = np.array([np.array([cur_vis[i], cur_vis[i]]) for i in range(0, len(cur_vis))])
                # #
                # cur_pts = cvt_pts(cur_pts)
                # cur_vis = cvt_pts(cur_vis)
                # cur_pts = cvt_pts(cur_pts)
                #
                # cur_pts = cvt256PtsTo94Pts(cur_pts)
                # cur_pts = cvt130PtsTo94Pts(cur_pts_p1)
                # merge -> 256
                cur_pts_p2 = np.concatenate((cur_pts_p2[0:16, :], cur_pts_p2[43:59, :], cur_pts_p2[16:40, :],
                                             cur_pts_p2[59:83, :], cur_pts_p2[86:158, :], cur_pts_p2[40:41, :],
                                             cur_pts_p2[83:84, :], cur_pts_p2[41:43, :], cur_pts_p2[84:86, :]), axis=0)
                cur_pts_p1 = np.squeeze(cur_pts_p1.reshape(117 * 2, 1))
                cur_pts_p2 = np.squeeze(cur_pts_p2.reshape(158 * 2, 1))
                face_pts = points_117_158_256(cur_pts_p2, cur_pts_p1)
                cur_pts = np.asarray(face_pts).reshape(int(len(face_pts) / 2), 2)


                pts_res_list.append(cur_pts)

        return pts_res_list, pts_score_list


def calc_iou(box1, box2):
    box1 = box1.copy()
    box2 = box2.copy()
    box1[2] = box1[0] + box1[2]
    box1[3] = box1[1] + box1[3]
    box2[2] = box2[0] + box2[2]
    box2[3] = box2[1] + box2[3]
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h<0 or in_w<0 else in_h*in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou

class AlignImage(object):
    def __init__(self, device='cuda', 
                 det_path='data/models/yoloface_v5l.pt', 
                 p1_path='data/models/p1.pt', 
                 p2_path='data/models/p2.pt',
                 pts217_path='data/models/res101_maxpool_pts217.bin'):
        self.facedet = YoloFace(pt_path=det_path, confThreshold=0.5, nmsThreshold=0.45, device=device)
        # self.align = RefinePts(device=device, p1_path=p1_path, p2_path=p2_path)
        if pts217_path.endswith('.pt'):
            self.pts217 = torch.jit.load(pts217_path).to(device).half()
        else:
            self.pts217 = LandmarkModel(pts217_path).to(device).half()
        
        expand_pad = 16
        self.mean_face_lm5p = np.array([
            [(30.2946+8)*2+16+expand_pad, 51.6963*2+expand_pad],  # left eye pupil
            [(65.5318+8)*2+16+expand_pad, 51.5014*2+expand_pad],  # right eye pupil
            [(48.0252+8)*2+16+expand_pad, 71.7366*2+expand_pad],  # nose tip
            [(33.5493+8)*2+16+expand_pad, 92.3655*2+expand_pad],  # left mouth corner
            [(62.7299+8)*2+16+expand_pad, 92.2041*2+expand_pad],  # right mouth corner
            ], dtype=np.float32)
        self.multipy_ratio = 256/(256+2*expand_pad)
        self.device = device

    def get_current_time(self):
        torch.cuda.synchronize()
        return time.time()

    @torch.no_grad()
    def align(self, frame, five_pts_list):
        pts_res_list = []
        for pts5 in five_pts_list:
            warp_mat = self.get_custom_affine_transform_expand(pts5.reshape(5,2))
            warp_mat_inverse=cv2.invertAffineTransform(warp_mat)
            crop_frame = cv2.warpAffine(frame, warp_mat, (256, 256), flags=cv2.INTER_LINEAR)
            crop_face_tensor = torch.from_numpy(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).to(self.device, non_blocking=False).half() / 127.5 - 1
            pts217 = self.pts217(crop_face_tensor).view(217,2).cpu().numpy()
            pts217_on_frame = np.dot(np.concatenate((pts217, np.ones((217,1), dtype=pts217.dtype)), axis=1), warp_mat_inverse.T)
            
            #twice
            pts5=self.get_pts5(pts217_on_frame)
            warp_mat = self.get_custom_affine_transform_expand(pts5.reshape(5,2))
            warp_mat_inverse=cv2.invertAffineTransform(warp_mat)
            crop_frame = cv2.warpAffine(frame, warp_mat, (256, 256), flags=cv2.INTER_LINEAR)
            crop_face_tensor = torch.from_numpy(cv2.cvtColor(crop_frame, cv2.COLOR_BGR2RGB)).permute(2,0,1).unsqueeze(0).to(self.device, non_blocking=False).half() / 127.5 - 1
            pts217 = self.pts217(crop_face_tensor).view(217,2).cpu().numpy()
            pts217_on_frame = np.dot(np.concatenate((pts217, np.ones((217,1), dtype=pts217.dtype)), axis=1), warp_mat_inverse.T)
            
            pts256=np.zeros((256,2), dtype=np.float32)
            pts256[0:215]=pts217_on_frame[0:215]
            pts256[222:224]=pts217_on_frame[215:217]
            pts_res_list.append(pts256)
        return pts_res_list
    
    
    def get_custom_affine_transform_expand(self, target_face_lm5p):
        mat_warp = np.zeros((2,3))
        A = np.zeros((4,4))
        B = np.zeros((4))
        for i in range(5):
            #sa[0][0] += a[i].x*a[i].x + a[i].y*a[i].y;
            A[0][0] += target_face_lm5p[i][0] * target_face_lm5p[i][0] + target_face_lm5p[i][1] * target_face_lm5p[i][1]
            #sa[0][2] += a[i].x;
            A[0][2] += target_face_lm5p[i][0]
            #sa[0][3] += a[i].y;
            A[0][3] += target_face_lm5p[i][1]

            #sb[0] += a[i].x*b[i].x + a[i].y*b[i].y;
            B[0] += target_face_lm5p[i][0] * self.mean_face_lm5p[i][0] * self.multipy_ratio + target_face_lm5p[i][1] * self.mean_face_lm5p[i][1] * self.multipy_ratio
            #sb[1] += a[i].x*b[i].y - a[i].y*b[i].x;
            B[1] += target_face_lm5p[i][0] * self.mean_face_lm5p[i][1] * self.multipy_ratio - target_face_lm5p[i][1] * self.mean_face_lm5p[i][0] * self.multipy_ratio
            #sb[2] += b[i].x;
            B[2] += self.mean_face_lm5p[i][0] * self.multipy_ratio
            #sb[3] += b[i].y;
            B[3] += self.mean_face_lm5p[i][1] * self.multipy_ratio

        #sa[1][1] = sa[0][0];
        A[1][1] = A[0][0]
        #sa[2][1] = sa[1][2] = -sa[0][3];
        A[2][1] = A[1][2] = -A[0][3]
        #sa[3][1] = sa[1][3] = sa[2][0] = sa[0][2];
        A[3][1] = A[1][3] = A[2][0] = A[0][2]
        #sa[2][2] = sa[3][3] = count;
        A[2][2] = A[3][3] = 5
        #sa[3][0] = sa[0][3];
        A[3][0] = A[0][3]

        _, mat23 = cv2.solve(A, B, flags=cv2.DECOMP_SVD)
        mat_warp[0][0] = mat23[0]
        mat_warp[1][1] = mat23[0]
        mat_warp[0][1] = -mat23[1]
        mat_warp[1][0] = mat23[1]
        mat_warp[0][2] = mat23[2]
        mat_warp[1][2] = mat23[3]
        return mat_warp

    def get_pts5(self, pts):
        if len(pts) == 5:
            fa5p = pts
        elif len(pts) == 90 or len(pts) == 94:
            fa5p = np.array([
                pts[16] * 0.5 + pts[20] * 0.5,
                pts[24] * 0.5 + pts[28] * 0.5,
                pts[32],
                pts[45],
                pts[51]], dtype=np.float32)
        elif len(pts) == 217 or len(pts) == 256:
            fa5p = np.array([
                pts[32] * 0.5 + pts[44] * 0.5,
                pts[56] * 0.5 + pts[68] * 0.5,
                pts[80],
                pts[102],
                pts[120]], dtype=np.float32)
        else:
            raise ValueError("[Error]Invalid Pts(%d)!" % len(pts))
        return fa5p
    
    @torch.no_grad()
    def __call__(self, im, maxface=False, ptstype='256', prev_bbox=None):
        # by default , face detection resize image height to 640
        # h,w,c= im.shape
        bboxes, kpss, scores = self.facedet.detect(im)
        face_num = bboxes.shape[0]
        # print('det face num : ', face_num)

        five_pts_list = []
        scores_list = []
        bboxes_list = []
        for i in range(face_num):
            five_pts_list.append(kpss[i].reshape(5,2))
            scores_list.append(scores[i])
            bboxes_list.append(bboxes[i])

        max_idx = -1
        if prev_bbox is not None and face_num > 1:
            max_iou = 0
            for i in range(face_num):
                iou = calc_iou(prev_bbox, bboxes_list[i])
                # print(iou)
                if iou > max_iou:
                    max_idx = i
                    max_iou = iou
            if max_idx >= 0:
                five_pts_list = [five_pts_list[max_idx]]
                scores_list = [scores_list[max_idx]]
                bboxes_list = [bboxes_list[max_idx]]
                
        if max_idx < 0 and maxface and face_num>1:
            max_idx = 0
            max_area = (bboxes[0, 2])*(bboxes[0, 3])
            for i in range(1, face_num):
                area = (bboxes[i,2])*(bboxes[i,3])
                if area>max_area:
                    max_idx = i
                    max_area = area
            five_pts_list = [five_pts_list[max_idx]]
            scores_list = [scores_list[max_idx]]
            bboxes_list = [bboxes_list[max_idx]]

        if ptstype=='5':
            return five_pts_list, scores_list, bboxes_list

        # ytpts_list, scores_list = self.align(im, five_pts_list)
        ytpts_list = self.align(im, five_pts_list)

        if ptstype=='94':
            pts94_list = [cvt256PtsTo94Pts(pts) for pts in ytpts_list]
            return pts94_list, scores_list, bboxes_list

        return ytpts_list, scores_list, bboxes_list



