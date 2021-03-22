import sys
import numpy as np
import os
import cv2
from tqdm import tqdm


caffe_root = '/home/liuky/HDD_1/workspace/C/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
import lmdb
from utils import base_util
from object_mark.config import ALL_CLASSES

# label_map = {'smoke': 1, 'mark': 2}


def datum_read():
    import lmdb
    import numpy as np
    import cv2
    from caffe.proto import caffe_pb2
    # /home/liuky/HDD_1/soft/caffe-ssd/examples/VOC0712/VOC0712_trainval_lmdb
    # /home/liuky/HDD_1/data/tmp_lmdb/smoke_data_lmdb_300
    lmdb_env = lmdb.open(
        '/home/liuky/HDD_1/data/tmp_lmdb/test_data')

    lmdb_txn = lmdb_env.begin()  # 生成处理句柄
    lmdb_cursor = lmdb_txn.cursor()  # 生成迭代器指针
    annotated_datum = caffe_pb2.AnnotatedDatum()  # AnnotatedDatum结构

    for idx,(key, value) in enumerate(lmdb_cursor):
        # print(idx)
        # continue

        annotated_datum.ParseFromString(value)
        datum = annotated_datum.datum  # Datum结构
        grps = annotated_datum.annotation_group  # AnnotationGroup结构，一个group表示一个lebel类，每个group下又有复数个annotation表示检测框box
        type = annotated_datum.type
        for grp in grps:
            label = grp.group_label
            for annotation in grp.annotation:
                instance_id = annotation.instance_id
                xmin = annotation.bbox.xmin * datum.width  # Annotation结构
                ymin = annotation.bbox.ymin * datum.height
                xmax = annotation.bbox.xmax * datum.width
                ymax = annotation.bbox.ymax * datum.height

        # Datum结构的label以及三个维度
        _ = datum.label  # 在目标检测的数据集中，这个label是没有意义的，真正的label是上面的group_label
        channels = datum.channels
        height = datum.height
        width = datum.width
        image_x = np.fromstring(datum.data, dtype=np.uint8)  # 字符串转换为矩阵

        if channels == 3:
            image = cv2.imdecode(image_x, -1)  # decode
            cv2.imwrite('/home/liuky/HDD_1/tmp.jpg', image)
        else:
            img_datas = []
            for channel_idx in range(channels):
                start_idx = channel_idx * height * width
                single_channel = image_x[start_idx:(start_idx + height * width)]
                single_channel = np.reshape(single_channel, (height, width))
                img_datas.append(single_channel[:, :, np.newaxis])
            image = np.concatenate(img_datas, -1)
            tmp_img = image[:,:,[0]]
            cv2.imwrite('/home/liuky/HDD_1/tmp.jpg', cv2.resize(tmp_img, (1280, 720)))

        print('')

        # cv2.imshow("image", image)  # 显示图片
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


def lmdb_building(root_folders, save_path):
    # label 格式 [x1,y1,x2,y2,instance_id,class]

    env = lmdb.open(save_path, map_size=1e12)  # 创建一个lmdb文件，map size表示这个文件的大小
    base_name = os.path.basename(save_path)
    if "test" in base_name:
        test_name_size_file_name = os.path.dirname(save_path) + '/test_name_size.txt'
        writer = open(test_name_size_file_name, 'w', encoding='utf-8')
    else: writer = None

    txn = env.begin(write=True)  # 调出指针，开始写数据
    # txn is a Transaction object
    lmdb_idx = 0
    for root_folder in root_folders:
        img_file_folder = os.path.join(root_folder, 'video_rgb12')
        target_file_folder = os.path.join(root_folder, 'all_mark_labels')
        image_file_list = base_util.get_file_list(img_file_folder, ['mp4'])
        print(root_folder)
        bar = tqdm(image_file_list)
        for i, image_path in enumerate(bar):
            # if i %100 ==99:
            #     break
            file_name = '.'.join(image_path.split('/')[-1].split('.')[:-1])
            if not writer is None:
                writer.write(file_name + ' 300 300 \n')
            image = base_util.data_read('video', image_path)
            image = [cv2.resize(image[idx], (300, 300))[...,np.newaxis] for idx in range(len(image))]
            if len(image)>12:
                image=[image[i] for i in range(1,len(image),2)]
            image = np.concatenate(image, -1)#[300,300,3,12]
            # image = np.random.randint(0,255,(480,640,36),np.uint8)
            h, w, channel,frame_num = image.shape
            target = base_util.load_label_file(os.path.join(target_file_folder, file_name + '.txt'), (1, 1))
            # target = np.array([[0,0,0.1,0.2,0,1]], np.float32)
            annotated_datum = caffe.proto.caffe_pb2.AnnotatedDatum()  # 声明一个AnnotatedDatum类对象
            annotated_datum.type = annotated_datum.BBOX

            for class_idx, class_str in enumerate(ALL_CLASSES):
                annotation_group = None
                instance_id = -1
                for box in target:  # 开始根据target构造box数据
                    (xmin, ymin, xmax, ymax, label_str) = box
                    if label_str == class_str:
                        instance_id += 1
                        if annotation_group == None:
                            annotation_group = annotated_datum.annotation_group.add()  # 声明一个新的annotation_group结构
                        label_num = class_idx
                        annotation_group.group_label = int(label_num)  # 传入label值
                        annotation = annotation_group.annotation.add()  # 声明一个annotation结构用来保存box数据
                        annotation.instance_id = instance_id  # 这个值表示这是当前图片下当前分类的第几个box,这里就先默认是第一个
                        annotation.bbox.xmin = float(xmin)
                        annotation.bbox.ymin = float(ymin)
                        annotation.bbox.xmax = float(xmax)
                        annotation.bbox.ymax = float(ymax)
                        annotation.bbox.difficult = False  # 表示是否是难识别对象，这里就默认不是好了


            datum = annotated_datum.datum  # 声明一个datum结构用于保存图像信息
            datum.channels = int(channel*frame_num)
            datum.height = h
            datum.width = w

            byte_data = bytes()
            for channel_idx in range(channel):
                single_channel = image[...,channel_idx,:]
                for frame_num_ in range(frame_num):
                    single_frame_num = single_channel[...,frame_num_]
                    # cv2.imwrite('/home/liuky/HDD_1/tmp.jpg', cv2.resize(single_channel[:,:,0], (1280, 720)))
                    byte_data += single_frame_num.tobytes()
            if len(byte_data)<5:
                raise Exception(image_path)
            datum.data = byte_data  # 将图像的array数组转为字节数据
            datum.encoded = False
            datum.label = -1  # 由于我们的label数据定义在annotation_group中了，所以这里默认为-1

            # The encode is only essential in Python 3
            txn.put(str(lmdb_idx).encode('ascii'), annotated_datum.SerializeToString())  # 保存annotated_datum到lmdb文件中
            lmdb_idx+=1
            bar.set_description("|file name: %s |w: %s|h: %s|c: %s|" %(file_name,str(w),str(h),str(channel)))
            if i % 499 == 0:
                txn.commit()
                txn = env.begin(write=True)
        txn.commit()
        txn = env.begin(write=True)
        print(env.stat())
    txn.commit()
    if not writer is None:
        writer.close()
    env.close()


if __name__ == '__main__':
    # '/home/liuky/HDD_1/data/smoke/train_data/1005smokedata/seq12_data/video_rgb12'
    folders = [
        '/home/liuky/HDD_1/data/smoke/train_data/youyi_plus/seq12_data_smoke',
        # '/home/liuky/HDD_1/data/smoke/train_data/youyi_plus/seq12_data',
        # '/home/liuky/HDD_1/data/smoke/train_data/youyi_rain_plus/seq12_data',
        # '/home/liuky/HDD_1/data/smoke/train_data/yunjin_plus/seq12_data',
        # '/home/liuky/HDD_1/data/smoke/train_data/1005smokedata/seq12_data',
        # '/home/liuky/HDD_1/data/smoke/train_data/heneng_plus/seq12_data',
    ]
    # '/home/liuky/HDD_1/data/smoke/train_data/1005smokedata/seq12_data/all_mark_labels'

    lmdb_building(folders, '/home/liuky/HDD_1/workspace/C/caffe/data/custom_obj_detect/test_data',)
    # datum_read()

    print('')
