import argparse

import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter

import test  # import test.py to get mAP after each epoch
from models import *
from utils.datasets import *
from utils.utils import *

mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex,关于分布式多线程
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')#windows版本还不是很成熟
    mixed_precision = False  # not installed

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last.pt'	#设置好检查点文件名字
best = wdir + 'best.pt'
results_file = 'results.txt'#数据结果名字

# Hyperparameters https://github.com/ultralytics/yolov3/issues/310
#原来是用的giou，现在全改为了ciou的变种
hyp = {'ciou': 3.54,  # ciou loss gain，iou的权重
       'cls': 37.4,  # cls loss gain，cls的权重
       'cls_pw': 1.0,  # cls BCELoss positive_weight，BCEWithLogitsLoss的参数
       'obj': 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)，obj的权重
       'obj_pw': 1.0,  # obj BCELoss positive_weight，BCEWithLogitsLoss的参数
       'iou_t': 0.20,  # iou training threshold
       'lr0': 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)#LR调度仅在纪元计数的80％和90％处生效。此后，mAP通常会显着增加。
       'lrf': 0.0005,  # final learning rate (with cos scheduler),余弦退火
       'momentum': 0.937,  # SGD momentum,
       'weight_decay': 0.000484,  # optimizer weight decay,权值衰减
       'fl_gamma': 0.0,  # focal loss gamma (efficientDet default is gamma=1.5),焦损失参数
       'hsv_h': 0.0138,  # image HSV-Hue augmentation (fraction)
       'hsv_s': 0.678,  # image HSV-Saturation augmentation (fraction)
       'hsv_v': 0.36,  # image HSV-Value augmentation (fraction)
       'degrees': 1.98 * 0,  # image rotation (+/- deg)
       'translate': 0.05 * 0,  # image translation (+/- fraction)
       'scale': 0.05 * 0,  # image scale (+/- gain)
       'shear': 0.641 * 0}  # image shear (+/- deg)

# Overwrite hyp with hyp*.txt (optional)，添加txt文件来方便更改超参数
f = glob.glob('hyp*.txt')#conda自带的库，返回与路径名模式匹配的列表
if f:
    print('Using %s' % f[0])
    for k, v in zip(hyp.keys(), np.loadtxt(f[0])):
        hyp[k] = v

# Print focal loss if gamma > 0,焦损失参数
if hyp['fl_gamma']:
    print('Using FocalLoss(gamma=%g)' % hyp['fl_gamma'])


def train():
    cfg = opt.cfg#从opt对象中拿到cfg文件路径
    data = opt.data#从opt对象中拿到data文件路径
    epochs = opt.epochs  # 500200 batches at bs 64, 117263 images = 273 epochs
    batch_size = opt.batch_size
    accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)，优化器的参数使用梯度累计64后更新
    weights = opt.weights  # initial training weights
    imgsz_min, imgsz_max, imgsz_test = opt.img_size  # img sizes (min, max, test)，最小320，最大640，测试的缺了，论文中是320-608

    # Image Sizes
    gs = 32  # (pixels) grid size，一个单元格占64个像素，需要改为32，后面注释我使用32
    assert math.fmod(imgsz_min, gs) == 0, '--img-size %g must be a %g-multiple' % (imgsz_min, gs)#要求图片像素大小是64的倍数？是不是网络变了？
    opt.multi_scale |= imgsz_min != imgsz_max  # multi if different (min, max)
    if opt.multi_scale:#如果设置了多尺度训练
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs#最大最小单元格个数设置为图片像素除以64，应该除以32,结果是10和19
        imgsz_min, imgsz_max = grid_min * gs, grid_max * gs#320,608
    img_size = imgsz_max  # initialize with max size，刚开始img_size初始化为我们设置的最大的尺寸608

    # Configure run
    init_seeds()#torch_utils.init_seeds来设置随机种子，固定为0
    ######################################################################################################################
    '''
    classes=80
    train=../coco/train2017.txt
    valid=../coco/val2017.txt
    names=data/coco.names
    '''
    data_dict = parse_data_cfg(data)#读取数据，默认为data/coco2017.data，返回字典
    train_path = data_dict['train']#训从上面的字典中拿到训练数据的路径
    test_path = data_dict['valid']#训从上面的字典中拿到验证数据的路径
    nc = 1 if opt.single_cls else int(data_dict['classes'])  # number of classes
    hyp['cls'] *= nc / 80  # update coco-tuned hyp['cls'] to current dataset，类损失增益，如果不是80类的话，相应处理一下

    # Remove previous results
    for f in glob.glob('*_batch*.jpg') + glob.glob(results_file):#列表相加直接合并，即先去除一个batch图片和result文件
        os.remove(f)
####################################################################################################################
    # Initialize model，在models里面的darknet类
    model = Darknet(cfg).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
########################需要把代码模型跑一遍，打印出来看一下
    for k, v in dict(model.named_parameters()).items():#model的每层参数的名称和参数值，named_parameters中是一个元组
        if '.bias' in k:#层名在字典参数的键名中
            pg2 += [v]  # biases，放到pg2中
        elif 'Conv2d.weight' in k:
            pg1 += [v]  # apply weight_decay，放到pg1中
        else:
            pg0 += [v]  # all else，其他参数放到pg0中,应该是shortcut层的权重,可能还会有别的
    
    if opt.adam:
        # hyp['lr0'] *= 0.1  # reduce lr (i.e. SGD=5E-3, Adam=5E-4)
        optimizer = optim.Adam(pg0, lr=hyp['lr0'])#初始学习率
        # optimizer = AdaBound(pg0, lr=hyp['lr0'], final_lr=0.1)
    else:
        optimizer = optim.SGD(pg0, lr=hyp['lr0'], momentum=hyp['momentum'], nesterov=True)#参数momentum=0.937，lr=0.01
    #在使用预训练权重时，增加一个参数组
    optimizer.add_param_group({'params': pg1, 'weight_decay': hyp['weight_decay']})  # add pg1 with weight_decay，卷积参数做衰减
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)，偏置参数
    del pg0, pg1, pg2

    start_epoch = 0
    best_fitness = 0.0
    attempt_download(weights)#如果指定路径中没有weights才会去下载
    if weights.endswith('.pt'):  # pytorch format
        # possible weights are '*.pt', 'yolov3-spp.pt', 'yolov3-tiny.pt' etc.
        chkpt = torch.load(weights, map_location=device)

        # load model
        try:
            chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
            model.load_state_dict(chkpt['model'], strict=False)
        except KeyError as e:
            s = "%s is not compatible with %s. Specify --weights '' or specify a --cfg compatible with %s. " \
                "See https://github.com/ultralytics/yolov3/issues/657" % (opt.weights, opt.cfg, opt.weights)
            raise KeyError(s) from e

        # load optimizer
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']

        # load results
        if chkpt.get('training_results') is not None:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt

        start_epoch = chkpt['epoch'] + 1
        del chkpt

    elif len(weights) > 0:  # darknet format
        # possible weights are '*.weights', 'yolov3-tiny.conv.15',  'darknet53.conv.74' etc.
        load_darknet_weights(model, weights)

    # Mixed precision training https://github.com/NVIDIA/apex，我不用这个
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    #lr = base_lr * lmbda(self.last_epoch)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)#将每个参数组的学习速率设置为给定函数的lf倍,如果是多个参数组那么lr_lambda是列表
    scheduler.last_epoch = start_epoch - 1  # see link below,last_epoch为开始的epoch的前一个epoch值
    # https://discuss.pytorch.org/t/a-problem-occured-when-resuming-an-optimizer/28822

    # Plot lr schedule
    # y = []
    # for _ in range(epochs):
    #     scheduler.step()
    #     y.append(optimizer.param_groups[0]['lr'])
    # plt.plot(y, '.-', label='LambdaLR')
    # plt.xlabel('epoch')
    # plt.ylabel('LR')
    # plt.tight_layout()
    # plt.savefig('LR.png', dpi=300)

    # Initialize distributed training，我的cpu只有一个，用不上
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.yolo_layers = model.module.yolo_layers  # move yolo layer indices to top level

    # LoadImagesAndLabels类继承自Dataset，dataset是数据集类，完成数据的增广
    # 其中__len__ 实现 len(dataset) 返还数据集中图片的个数，__getitem__ 用来获取一些索引数据，例如使用dataset[i] 获得第i个样本
    # img_size先初始化为608，最大号，batch_size=16(argparse.ArgumentParser()中改)
    # opt.cache_images缓存图片以进行更快地训练
    # opt.single_cls单类
    # __getitem__会返回3*608*608的数据，n行6列的标签(,类别,归一化的x0,归一化的y0,归一化的x1,归一化的y1)，图片的索引，shapes=none
    dataset = LoadImagesAndLabels(train_path, img_size, batch_size,#这里的batch貌似没有用到啊
                                  augment=True,
                                  hyp=hyp,  # augmentation hyperparameters，数组增广超参数
                                  rect=opt.rect,  # rectangular training，矩形训练
                                  cache_images=opt.cache_images,#缓存图片以进行更快地训练
                                  single_cls=opt.single_cls)             #没有启用不同数量目标损失调整

    # Dataloader
    batch_size = min(batch_size, len(dataset))#16，dateset中设置的是16
    nw = min([4, batch_size if batch_size > 1 else 0, 8])  # number of workers[2,16,8]，研究室电脑可以把2改为4,4线程
    # dataloader是一个迭代器，里面有每一批的数据，是字典，字典中有image(储存图片数据)+landmarks(所有标签数据)，在dataset上实现batch
    # len(dataloader)是一批数据，与len(dataset)不一样
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Testloader,test是也是进行数据增广的！，但没有用mosic，而是开启了rect
    testloader = torch.utils.data.DataLoader(LoadImagesAndLabels(test_path, imgsz_test, batch_size,
                                                                 hyp=hyp,
                                                                 rect=True,
                                                                 cache_images=opt.cache_images,
                                                                 single_cls=opt.single_cls),
                                             batch_size=batch_size,
                                             num_workers=nw,
                                             pin_memory=True,
                                             collate_fn=dataset.collate_fn)

    # Model parameters
    model.nc = nc  # attach number of classes to model 80
    model.hyp = hyp  # attach hyperparameters to model 在26行
    model.gr = 1.0  # giou loss ratio (obj_loss = 1.0 or giou)giou的损失比率，初始化，后面会改
    #dataset.labels：[[5列n行],[5列n行],[5列n行],[5列n行]]，每一行代表一个物体，第一列是标签类别，后4列是对角坐标
    model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device)  # attach class weights,不同类别的权重

    # Model EMA
    ema = torch_utils.ModelEMA(model)

    # Start training
    nb = len(dataloader)  # number of batches，我要设置为8
    n_burn = max(3 * nb, 500)  # burn-in iterations, max(3 epochs, 500 iterations)，500
    maps = np.zeros(nc)  # mAP per class，初始化为80个0【0,0,0...】
    # torch.autograd.set_detect_anomaly(True)
    results = (0, 0, 0, 0, 0, 0, 0)  # 'P', 'R', 'mAP', 'F1', 'val GIoU', 'val Objectness', 'val Classification'
    t0 = time.time()
    print('Image sizes %g - %g train, %g test' % (imgsz_min, imgsz_max, imgsz_test))
    print('Using %g dataloader workers' % nw)#我使用4线程
    print('Starting training for %g epochs...' % epochs)#迭代多少个epochs
    # start_epoch默认从0开始，如果恢复训练的话会从weights文件中拿到epoch+1
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()

        # Update image weights (optional)没有使用
        if dataset.image_weights:
            w = model.class_weights.cpu().numpy() * (1 - maps) ** 2  # class weights
            image_weights = labels_to_image_weights(dataset.labels, nc=nc, class_weights=w)
            dataset.indices = random.choices(range(dataset.n), weights=image_weights, k=dataset.n)  # rand weighted idx

        mloss = torch.zeros(4).to(device)  # mean losses[0,0,0,0]
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem', 'IoU', 'obj', 'cls', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(dataloader), total=nb)  # progress bar
        # targets = [一批中的标签的图片索引, class, x, y, w, h]
        for i, (imgs, targets, paths, _) in pbar:  # batch -------------------------------------------------------------
            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0，像素值也归一化了
            targets = targets.to(device) #n行6列的标签(,类别,归一x中心点，归一y中心点,归一宽总度，归一总高度)，图片的索引，shapes=none

            # Burn-in,上面的学习率是对epoch进行，这里是在一个epoch内进行
            if ni <= n_burn * 2:#ni是迭代次数，如果小于500*2的话,根据迭代次数和epoch数对学习率(包括其momentum)进行cos退火
                #做插值，将ni从[0,500]映射到[0,1]的数，赋值为gr
                model.gr = np.interp(ni, [0, n_burn * 2], [0.0, 1.0])  # giou loss ratio (obj_loss = 1.0 or giou)
                if ni == n_burn:  # burnin complete
                    print_model_biases(model)

                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    # 学习率从根据迭代次数，在[0.1/0,0.01*退火]进行映射
                    x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, [0, n_burn], [0.9, hyp['momentum']])#动量从根据迭代次数,在[0.9,0.937上映射]

            # Multi-Scale
            if opt.multi_scale:
                if ni / accumulate % 1 == 0:  #  adjust img_size (67% - 150%) every 1 batch
                    img_size = random.randrange(grid_min, grid_max + 1) * gs#(10,19)*32
                sf = img_size / max(imgs.shape[2:])  # scale factor
                if sf != 1:
                    ns = [math.ceil(x * sf / gs) * gs for x in imgs.shape[2:]]  # new shape (stretched to 32-multiple)
                    imgs = F.interpolate(imgs, size=ns, mode='bilinear', align_corners=False)

            # Forward
            pred = model(imgs)

            # Loss
            loss, loss_items = compute_loss(pred, targets, model)#lss是一个数，loss_items=tensor([lbox, lobj, lcls, loss])
            if not torch.isfinite(loss):#是不是无穷大
                print('WARNING: non-finite loss, ending training ', loss_items)
                return results

            # Backward
            loss *= batch_size / 64  # scale loss,
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            # Optimize
            if ni % accumulate == 0:#如果迭代次数是64的倍数，就更新参数，类似于把batch设为64的效果
                optimizer.step()
                optimizer.zero_grad()
                ema.update(model)

            # Print
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = '%.3gG' % (torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
            s = ('%10s' * 2 + '%10.3g' * 6) % ('%g/%g' % (epoch, epochs - 1), mem, *mloss, len(targets), img_size)#每一批的结果
            pbar.set_description(s)

            # Plot
            if ni < 1:#即刚开始时，没啥用
                f = 'train_batch%g.jpg' % i  # filename
                plot_images(imgs=imgs, targets=targets, paths=paths, fname=f)
                if tb_writer:
                    tb_writer.add_image(f, cv2.imread(f)[:, :, ::-1], dataformats='HWC')
                    # tb_writer.add_graph(model, imgs)  # add model to tensorboard

            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler，按每个批次更新
        scheduler.step()

        # Process epoch results
        ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        # not>and>or
        if not opt.notest or final_epoch:  # Calculate mAP，如果测试每一批的结果的epoch或者达到了最后的epoch
            is_coco = any([x in data for x in ['coco.data', 'coco2014.data', 'coco2017.data']]) and model.nc == 80
            #save_json仅在最后一个epoches时为True
            '''data
            classes=80
            train=../coco/trainvalno5k.txt
            valid=../coco/5k.txt
            names=data/coco.names
            '''
            results, maps = test.test(cfg,
                                      data,
                                      batch_size=batch_size,
                                      img_size=imgsz_test,
                                      model=ema.ema,
                                      save_json=final_epoch and is_coco,
                                      single_cls=opt.single_cls,
                                      dataloader=testloader)

        # Write epoch results
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 7 % results + '\n')  
            # epoch,mem, mloss, len(targets), img_size),P, R, mAP, F1, test_losses=(GIoU, obj, cls)
        if len(opt.name) and opt.bucket:#如果重命名文件或者..
            os.system('gsutil cp results.txt gs://%s/results/results%s.txt' % (opt.bucket, opt.name))

        # Write Tensorboard results
        if tb_writer:#默认设置了None
            tags = ['train/giou_loss', 'train/obj_loss', 'train/cls_loss',
                    'metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/F1',
                    'val/giou_loss', 'val/obj_loss', 'val/cls_loss']
            for x, tag in zip(list(mloss[:-1]) + list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)

        # Update best mAP
        fi = fitness(np.array(results).reshape(1, -1))  # fitness_i = weighted combination of [P, R, mAP, F1]
        if fi > best_fitness:
            best_fitness = fi

        # Save training results，保存训练的chkpt检查点文件
        save = (not opt.nosave) or (final_epoch and not opt.evolve)#默认nosave=False，只保存最后的检查文件
        if save:
            with open(results_file, 'r') as f:
                # Create checkpoint
                chkpt = {'epoch': epoch,
                         'best_fitness': best_fitness,
                         'training_results': f.read(),
                         'model': ema.ema.module.state_dict() if hasattr(model, 'module') else ema.ema.state_dict(),
                         'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last checkpoint
            torch.save(chkpt, last)#保存最后的检查点文件,last是保存的路径(包含文件名)

            # Save best checkpoint
            if (best_fitness == fi) and not final_epoch:
                torch.save(chkpt, best)

            # Save backup every 10 epochs (optional)
            # if epoch > 0 and epoch % 10 == 0:
            #     torch.save(chkpt, wdir + 'backup%g.pt' % epoch)

            # Delete checkpoint
            del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    # end training
    n = opt.name
    if len(n):
        n = '_' + n if not n.isnumeric() else n
        fresults, flast, fbest = 'results%s.txt' % n, wdir + 'last%s.pt' % n, wdir + 'best%s.pt' % n
        for f1, f2 in zip([wdir + 'last.pt', wdir + 'best.pt', 'results.txt'], [flast, fbest, fresults]):
            if os.path.exists(f1):
                os.rename(f1, f2)  # rename
                ispt = f2.endswith('.pt')  # is *.pt
                strip_optimizer(f2) if ispt else None  # strip optimizer
                os.system('gsutil cp %s gs://%s/weights' % (f2, opt.bucket)) if opt.bucket and ispt else None  # upload

    if not opt.evolve:
        plot_results()  # save as results.png,绘制图片
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()

    return results


if __name__ == '__main__':
    #https://blog.csdn.net/The_Time_Runner/article/details/97941409，argparse命令行接口使用，方面在cmd或shell直接添加参数
    parser = argparse.ArgumentParser()#实例化
    parser.add_argument('--epochs', type=int, default=5)  # 500200 batches at bs 16, 117263 COCO images = 273 epochs
    parser.add_argument('--batch-size', type=int, default=8)  # effective bs = batch_size * accumulate = 16 * 4 = 64
    parser.add_argument('--cfg', type=str, default='cfg/rfb-yolov3', help='*.cfg path')
    parser.add_argument('--data', type=str, default='data/coco.data', help='*.data path')#里面是具体的文件路径
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67%% - 150%%) img_size every 10 batches')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320,608,512],
                        help='[min_train, max-train, test] img sizes')
    parser.add_argument('--rect', action='store_true', help='rectangular training')#与mosaic对立，设置为true后不使用mosaic
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')#不实用
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='initial weights path')
    parser.add_argument('--name', default='', help='renames results.txt to results_name.txt if supplied')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    opt = parser.parse_args()#解析参数
    opt.weights = last if opt.resume else opt.weights#如果设置了resume的话，则网络的权重文件名为 wdir + 'last.pt'
    #check_git_status()#看当前git的情况
    print(opt)
    opt.img_size.extend([opt.img_size[-1]] * (3 - len(opt.img_size)))  # extend to 3 sizes (min, max, test)
    device = torch_utils.select_device(opt.device, apex=mixed_precision, batch_size=opt.batch_size)
    if device.type == 'cpu':
        mixed_precision = False

    # scale hyp['obj'] by img_size (evolved at 320)
    # hyp['obj'] *= opt.img_size[0] / 320.

    tb_writer = None#先是None初始化，后面改为True了
    if not opt.evolve:  # Train normally
        print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
        tb_writer = SummaryWriter(comment=opt.name)
        train()  # train normally

    else:  # Evolve hyperparameters (optional)演化超参数
        opt.notest, opt.nosave = True, True  # only test/save final epoch
        if opt.bucket:
            os.system('gsutil cp gs://%s/evolve.txt .' % opt.bucket)  # download evolve.txt if exists

        for _ in range(1):  # generations to evolve
            if os.path.exists('evolve.txt'):  # if evolve.txt exists: select best hyps and mutate
                # Select parent(s)
                parent = 'single'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt('evolve.txt', ndmin=2)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min()  # weights
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                method, mp, s = 3, 0.9, 0.2  # method, mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([1, 1, 1, 1, 1, 1, 1, 0, .1, 1, 0, 1, 1, 1, 1, 1, 1, 1])  # gains
                ng = len(g)
                if method == 1:
                    v = (npr.randn(ng) * npr.random() * g * s + 1) ** 2.0
                elif method == 2:
                    v = (npr.randn(ng) * npr.random(ng) * g * s + 1) ** 2.0
                elif method == 3:
                    v = np.ones(ng)
                    while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                        # v = (g * (npr.random(ng) < mp) * npr.randn(ng) * s + 1) ** 2.0
                        v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = x[i + 7] * v[i]  # mutate

            # Clip to limits
            keys = ['lr0', 'iou_t', 'momentum', 'weight_decay', 'hsv_s', 'hsv_v', 'translate', 'scale', 'fl_gamma']
            limits = [(1e-5, 1e-2), (0.00, 0.70), (0.60, 0.98), (0, 0.001), (0, .9), (0, .9), (0, .9), (0, .9), (0, 3)]
            for k, v in zip(keys, limits):
                hyp[k] = np.clip(hyp[k], v[0], v[1])

            # Train mutation
            results = train()

            # Write mutation results
            print_mutation(hyp, results, opt.bucket)

            # Plot results
            # plot_evolution_results(hyp)
