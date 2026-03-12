# Tài Liệu Kỹ Thuật TranSTR-Causal (Bản DINOv3 + Grounded-SAM)

Tài liệu này mô tả kiến trúc và chuẩn dữ liệu đầu vào sau khi thay đổi pipeline:
- Bo VIT, thay bang feature frame tu DINOv3.
- Object su dung Grounded-SAM (grounded theo question prompt).
- Bo buoc top-K object trong model, object da duoc lay truc tiep theo prompt.

Muc tieu la de ban lap ke hoach sua code theo tung buoc, tranh lech shape, lech toa do box, va lech frame index.

## 1. Tong quan kien truc moi

Luong xu ly de xuat:
1. Lay mau 16 frame/video (giong pipeline cu).
2. Trich xuat frame feature bang DINOv3 cho 16 frame.
3. Text encoder ma hoa question.
4. Frame decoder + frame top-K (giu nguyen co che loc frame theo question).
5. Grounded-SAM detection theo question prompt de lay object boxes/labels/scores.
6. ROI feature duoc cat tu frame feature map theo boxes (khong dung obj top-K nua).
7. Frame-object fusion.
8. Vision-language fusion.
9. Answer decoder va classifier.

Ghi chu:
- Ban co the chay Grounded-SAM tren toan bo 6 cau hoi cua cung video, sau do luu rieng ket qua theo tung qtype.
- Trong forward, van xu ly tung sample (video_id + question_type) nhu hien tai.

## 2. Dinh nghia input moi va shape bat buoc

### 2.1 Frame feature tu DINOv3

Ban can xac dinh ro ban dung 1 trong 2 che do:

1. Global frame token
- Shape moi frame: [D_frame]
- Sau stack 16 frame: [16, D_frame]
- Batch: [B, 16, D_frame]

2. Spatial feature map (khuyen nghi de cat ROI)
- Shape moi frame: [C, Hf, Wf]
- Sau stack 16 frame: [16, C, Hf, Wf]
- Batch: [B, 16, C, Hf, Wf]

Khuyen nghi:
- Neu muon ROIAlign that su theo box, ban nen luu spatial feature map.
- Neu chi co global token [B,16,D], ban khong cat ROI dung nghia duoc.

### 2.2 Object input tu Grounded-SAM

Moi frame cho moi question can co:
- boxes_xyxy_norm: [N, 4] (da normalize [0,1])
- scores: [N]
- labels_id hoac labels_text: [N]
- valid_mask: [N] (1 la object hop le, 0 la pad)

Sau khi pad den N_max/object per frame:
- boxes: [B, 16, N_max, 4]
- scores: [B, 16, N_max]
- labels: [B, 16, N_max]
- obj_mask: [B, 16, N_max]

Neu frame da qua top-K frame (Kf):
- boxes_sel: [B, Kf, N_max, 4]
- scores_sel: [B, Kf, N_max]
- labels_sel: [B, Kf, N_max]
- obj_mask_sel: [B, Kf, N_max]

### 2.3 ROI feature va object token

Neu co spatial feature map:
- ROIAlign input: frame_map [B*Kf, C, Hf, Wf], boxes tren cung he toa do.
- Roi pooled per object: [B, Kf, N_max, C_roi]

Object token de dua vao decoder:
- geom_embed(box + area + aspect): [B, Kf, N_max, d_model]
- label_embed: [B, Kf, N_max, d_model]
- score_embed: [B, Kf, N_max, d_model]
- roi_proj: [B, Kf, N_max, d_model]

Tong hop:
- obj_token = LN(roi_proj + geom_embed + label_embed + score_embed)
- Shape cuoi: [B, Kf, N_max, d_model]

## 3. Quy tac dong bo khong gian (rat quan trong)

Ban da bo buoc hop ly hoa khong gian theo kieu cu, nhung van phai dong bo toa do box va feature map bang mot quy uoc duy nhat.

Quy tac de xai on dinh:
1. Luu box theo toa do normalize [0,1] tren khong gian anh dau vao detector/backbone sau preprocess.
2. Luu metadata transform:
   - original_h, original_w
   - model_input_h, model_input_w
   - resize_mode (stretch/letterbox)
   - scale, pad_x, pad_y neu letterbox
3. Khi cat ROI tren frame feature map, phai map box sang dung he toa do cua map do.

Neu detector va encoder preprocess khac nhau:
- Bat buoc co ham quy doi box detector -> encoder map.
- Neu khong, ROI se lech va attention object-frame bi hu.

## 4. Luu y ve "detect tren ca 6 cau hoi"

Ban co the detect truoc cho ca 6 qtype cua cung video, nhung can luu rieng theo key:
- key de xuat: video_id + qtype

Khong nen dung chung mot bo object cho tat ca qtype vi:
- Cau hoi descriptive/explanatory/reasoning nhin doi tuong khac nhau.
- Neu dung chung, grounding theo prompt se mat y nghia.

## 5. Mapping sua code trong model/network

### 5.1 DataLoader

Can cap nhat output __getitem__:
1. frame_feat_dinov3
2. object package moi (boxes, scores, labels, obj_mask)
3. question va answer nhu cu

Shape tra ve de xai on dinh:
- frame_feat: [16, D] hoac [16, C, Hf, Wf]
- boxes: [16, N_max, 4]
- scores: [16, N_max]
- labels: [16, N_max]
- obj_mask: [16, N_max]

### 5.2 networks/model.py

Buoc giu nguyen:
- text encoder
- frame decoder + frame top-K
- vl encoder + ans decoder

Buoc thay doi:
1. Bo obj_sorter va bo nhanh top-K object.
2. Them object encoder moi:
   - box positional embedding
   - label embedding
   - score projection
   - roi projection (neu co ROI feature)
3. Tao obj_token tu object package da grounded.
4. Gather object theo idx_frame (neu co top-K frame).
5. Dua obj_token vao fo_decoder de fuse voi frame_local.

### 5.3 networks modules nen tach rieng

De de maintain, nen them file module rieng:
- object_encoder_grounded.py

Noi dung module:
1. build_object_tokens
2. roi_align_wrapper (neu co map)
3. box_transform_utils

## 6. Shape contract de test nhanh (checkpoint truoc khi train)

Truoc khi train full, can assert shape trong forward:
1. q_local: [B, Lq, d_model]
2. frame_local_raw: [B, Kf, d_model]
3. obj_token: [B, Kf, N_max, d_model]
4. frame_obj sau fo_decoder: [B, Kf, d_model]
5. mem: [B, Kf + Lq, d_model]
6. logits: [B, 5]

Neu roi map duoc dung:
- roi_feat: [B, Kf, N_max, C_roi]

## 7. Ke hoach sua code tung buoc (planning checklist)

### Step 1 - Chot format feature va metadata
1. Chot DINOv3 output: global hay spatial map.
2. Chot N_max object/frame.
3. Chot format luu boxes/scores/labels/mask/metadata transform.

### Step 2 - Sua DataLoader
1. Load frame feature moi.
2. Load grounded object package theo key video_id + qtype.
3. Pad object den N_max va tao obj_mask.
4. Them unit test nho cho shape.

### Step 3 - Sua model forward
1. Giu frame selection.
2. Gather object theo idx_frame.
3. Build obj_token moi, bo obj_topK branch cu.
4. Fuse frame-object va chay tiep vl + answer decoder.

### Step 4 - Kiem tra alignment
1. Ve debug box len frame goc.
2. Ve debug box sau quy doi len feature map.
3. Kiem tra object mask co pad dung.

### Step 5 - Train baseline
1. Chay voi box + label + score (chua ROI) de verify pipeline.
2. Sau do mo ROI feature de tang hieu qua.
3. So sanh voi baseline cu tren d/e/par/car/all.

## 8. Rui ro thuong gap va cach tranh

1. Lech frame index giua frame feature va object file
- Cach tranh: luu frame_idx goc, assert trung khi load.

2. Lech toa do do resize mode khac nhau
- Cach tranh: luu metadata transform day du, test visual overlay.

3. Object qua nhieu lam no bo nho va loang attention
- Cach tranh: gioi han N_max, dung score threshold.

4. Missing object o mot so frame
- Cach tranh: pad zero token + obj_mask = 0.

## 9. Tom tat quyet dinh kien truc

1. Frame: DINOv3, van sample 16 frame nhu pipeline cu.
2. Object: Grounded-SAM theo question prompt.
3. Bo top-K object trong model, dung truc tiep object grounded.
4. Van nen giu top-K frame de toi uu compute va giam nhieu.
5. Neu can ROI that su, bat buoc luu/giu spatial feature map tu frame encoder.
