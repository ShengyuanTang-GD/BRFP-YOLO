import argparse
import json

from torch.utils.data import DataLoader

from models import *
from utils.datasets import *
from utils.utils import *

'''data
   classes=80
   train=../coco/trainvalno5k.txt
   valid=../coco/5k.txt
   names=data/coco.names
'''
#训练时调用和测试时调用
def test(cfg,
         data,
         weights=None,
         batch_size=16,
         img_size=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=False,# 最后一个批次时=True
         single_cls=False,
         augment=False,#默认是flase
         model=None,
         dataloader=None):
    # Initialize/load model and set device
    if model is None:
        device = torch_utils.select_device(opt.device, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, img_size)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            load_darknet_weights(model, weights)

        # Fuse
        model.fuse()
        model.to(device)

        if device.type != 'cpu' and torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    else:  # called by train.py
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)#返回一个字典{'classes':'80','train':'../coco/trainvalno5k.txt','valid':'../coco/5k.txt','names':'data/coco.names'}
    nc = 1 if single_cls else int(data['classes'])  # number of classes,80
    path = data['valid']  # path to test images,'../coco/5k.txt'
    names = load_classes(data['names'])  # class names,'data/coco.names'
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95[0.5到0.95的10个数]
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95,这里iouv=[0.5],这里只测了AP50！！！！！！！！！！！！！！
    niou = iouv.numel()#1

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, img_size, batch_size, rect=True, single_cls=opt.single_cls)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    seen = 0#记录图片个数
    model.eval()#模型进入测试模式
    _ = model(torch.zeros((1, 3, img_size, img_size), device=device)) if device.type != 'cpu' else None  # run once
    coco91class = coco80_to_coco91_class()# converts 80-index (val2014) to 91-index (paper)
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    p, r, f1, mp, mr, map, mf1, t0, t1 = 0., 0., 0., 0., 0., 0., 0., 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    # imgs是图片数据
    # targets = [一批中的图片索引号, class, x, y, w, h]
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width，先h后w是cv中的坐标结构
        whwh = torch.Tensor([width, height, width, height]).to(device)#先w后h是模型中的坐标结构

        # Plot images with bounding boxes
        f = 'test_batch%g.jpg' % batch_i  # filename
        if batch_i < 1 and not os.path.exists(f):
            plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            #inf_out是[8, 1521, 85],经[8,507,85]左右拼接
            inf_out, train_out = model(imgs, augment=augment)  # inference and training outputs,看models.pyde 310行
            t0 += torch_utils.time_synchronized() - t#跑模型阶段

            # Compute loss
            if hasattr(model, 'hyp'):  # if model has loss hyperparameters
                # train_out是3个(batch,3,13,13,85)
                loss += compute_loss(train_out, targets, model)[1][:3]  # IoU, obj, cls

            # Run NMS
            t = torch_utils.time_synchronized()
            #out是一个8个元素的列表，列表中每个元素是多行6列的box信息(归一x1,归一y1,归一x2,归一y2,cls*obj,cls_idx)
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres)  # nms
            t1 += torch_utils.time_synchronized() - t#nms阶段

        # Statistics per image
        for si, pred in enumerate(output):
            # targets = 多行*(一批中的图片索引号, class, x, y, w, h)
            labels = targets[targets[:, 0] == si, 1:]#取出标签中的Img_id和si对应上的标签，即同一张图片的标签,多行,索引号不要了
            nl = len(labels)#n个目标
            tcls = labels[:, 0].tolist() if nl else []  # target class，如果有目标,tcls=clsss标签类别
            seen += 1#记录图片个数

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            # 将归一化后的x_w1,y_h1,x_w2,y_h2变为原来的尺度
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary，最后一个epoch时save_json=True
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            #(m,1),m表示NMS后保留的box数量
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)#用于记录哪个box是正例
            if nl:#该图片中有目标
                detected = []  # target indices
                tcls_tensor = labels[:, 0]#类别,m行1列

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh#把标签的坐标形式从x1中心点,y1中心点,w,h变为x_w1,y_h1,x_w2,y_h2，并从归一化后的尺度还原到图片尺度
                #至此，预测结果和标签结果都统一到了原图片初度，且坐标形式为x_w1,y_h1,x_w2,y_h2
                # Per target class
                for cls in torch.unique(tcls_tensor):#去重后，遍历每一个标签类别，不妨是狗
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices，ti是标签结果中狗的index，一行
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices，pi是预测结果中狗的index，一行

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        '''
                            box_iou的Arguments:
                                box1 (Tensor[N, 4])#预测的
                                box2 (Tensor[M, 4])#标签的
                            Returns:
                                    iou (Tensor[N, M]):
                        '''
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices，max按行返回最大值和列索引(一行)

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():#如果>0.5，即算ap50
                            d = ti[i[j]]  # detected target，与所有预测框的Iou大于阈值的标签框的索引值
                            if d not in detected:
                                detected.append(d)#检测出来的标签的索引值，即labels的第几行
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn，把correct中的第pi[j]个索引改为true
                                if len(detected) == nl:  # all targets already located in image，如果iou大于阈值，说明被某个预测框检测到了，就放到d中
                                    break
                                
            # Append statistics (correct, conf, pcls, tcls)，每个图片遍历结束后把结果都放到stats列表中
            # correct是预测结果的正例bool
            #  pred是某张图片的6列的box信息(原x_w1,原y_h1,原x_w2,原y_h2,cls*obj,cls_idx)
            # tcls是某张图片的所有标签
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))#seen表示总目标个数

    # Print results per class
    if verbose and nc > 1 and len(stats):#verbose在训练时不使用，单独运行test时使用,其作用是一些描述
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (img_size, img_size, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and map and len(jdict):
        print('\nCOCO mAP with pycocotools...')
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
        except:
            print('WARNING: missing pycocotools package, can not compute official COCO mAP. See requirements.txt.')

        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        cocoGt = COCO(glob.glob('../coco/annotations/instances_val*.json')[0])  # initialize COCO ground truth api
        cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')#'box'表示目标检测，此外还有关键点检测和语义检测
        cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
        #在给定图像上运行每个图像评估，并将结果（字典列表）存储在self.evalImgs中：返回：无
        cocoEval.evaluate()
        #累积每个图像评估结果并将结果存储在self.eval：param p：用于评估的输入参数：return：无
        cocoEval.accumulate()
        #计算并显示用于评估结果的摘要指标
        #请注意，此功能只能*仅*应用于默认参数设置
        cocoEval.summarize()
        # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps


if __name__ == '__main__':#如果单独运行该test.py
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--cfg', type=str, default='cfg/rfb-yolov3', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco2014.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weight', help='weights path')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    print(opt)

    # task = 'test', 'study', 'benchmark'
    if opt.task == 'test':  # (default) test normally
        test(opt.cfg,
             opt.data,
             opt.weights,
             opt.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment)
