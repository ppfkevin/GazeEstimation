import os
import cv2
from interval import Interval
from Collect_Infor import Collect_information1
from Excel_read import read_file,filter_excel,find_index,valid_frames

def Process2(k):
    #6.089s:60
    valid_time = 6.138      #有效帧对应时间戳
    valid_frame = 98        #同步有效帧
    start_time = 10*60 + 26.858        #同步视频开始时间
    finish_time = 10*60 + 36.869    #同步视频结束时间
    synchronous_time = round((1 - valid_frame/600) * (finish_time-start_time) + valid_time, 3)    #同步视频播放结束对应时间戳

    discrete_headpose_start = - finish_time + synchronous_time + 10*60 + 42.845
    discrete_headpose_end = - finish_time + synchronous_time + 13*60 + 12.827
    section3 = Interval(discrete_headpose_start,discrete_headpose_end)
    continus_headpose_start = - finish_time + synchronous_time + 13*60 + 28.697
    continus_headpose_end = - finish_time + synchronous_time + 13*60 + 50.597
    section4 = Interval(continus_headpose_start,continus_headpose_end)
    Sections = [section3, section4]
    video_end = continus_headpose_end

    tobii_discrete_headpose_start = 10*60 + 42.845
    tobii_discrete_headpose_end = 13*60 + 12.827
    tobii_section3 = Interval(tobii_discrete_headpose_start, tobii_discrete_headpose_end)
    tobii_continus_headpose_start = 13*60 + 28.697
    tobii_continus_headpose_end = 13*60 + 50.597
    tobii_section4 = Interval(tobii_continus_headpose_start, tobii_continus_headpose_end)
    tobii_Sections = [tobii_section3, tobii_section4]

    root_path = 'F:\images\SJTUGaze\Pang_data\P16'

    gopro_paths = [os.path.join(root_path,'Eyetracking\GP1'), os.path.join(root_path,'Eyetracking\GP2'), os.path.join(root_path,'Eyetracking\GP3'), os.path.join(root_path,'Eyetracking\GP4')]
    gopro_path = gopro_paths[k]
    person = 'p016_'
    images_path = os.path.join(gopro_path,'Samples2')
    if not os.path.exists(images_path):
        os.mkdir(images_path)
    file = read_file(os.path.join(root_path,'Zhengchuanyang_New test_wushun2_yes_wushun2.xlsx'))
    tot_list, time_list = filter_excel(file)                       #excel表格中所有元素汇集
    videos_name = ['GOPR0576.MP4', 'GOPR0418.MP4', 'GOPR0275.MP4', 'GOPR0389.MP4']
    cap = cv2.VideoCapture(os.path.join(gopro_path, videos_name[k]))

    fps = cap.get(cv2.CAP_PROP_FPS)

    valid_tot = valid_frames(tot_list, tobii_Sections)       #valid_tot代表全部有效帧(0/4,4/0,0/0)
    sampling_rate = 1000/valid_tot
    sampling_interval = round(1/sampling_rate)
    cnt = 0         #计数所有读到的视频帧
    flag = False
    serial = 0      #计数所有有效帧
    serial2 = 0     #images路径下图片序号


    with open(os.path.join(images_path, 'annotation1.txt'), 'w') as W:
        with open(os.path.join(images_path, 'annotation2.txt'), 'w') as W2:
            while 1:
                ret,frame = cap.read()
                if not ret:
                    # if cnt > fps*video_end - 10000:
                    if cnt > 10000:
                        print('no frame read')
                        print(cnt)
                        break
                cnt += 1
                time = round(cnt/fps, 3)
                for section in Sections:
                    if time in section:
                        tobii_time = round(time - section.lower_bound + tobii_Sections[Sections.index(section)].lower_bound, 3)
                        Ind = find_index(time_list, tobii_time)
                        flag = True
                        # print('Index is:%d'%Ind)
                if flag:
                    serial += 1
                    if serial%sampling_interval == 0:
                        if Ind == 0:
                            break
                        if tot_list[Ind]['ValidityLeft']*tot_list[Ind]['ValidityRight'] == 0:
                            serial2 += 1
                            collect_name = person + str(serial2) + '.jpg'
                            Collect_information1(Ind, tot_list, W)
                            Collect_information1(Ind, tot_list, W2)
                            cv2.imwrite(os.path.join(images_path,collect_name), frame)
                            print(collect_name)
                flag = False
                # print(time)
                # print(cnt)

    print('Finished:%d'%(k+1))
    cap.release()
    if k==3:
        return None
    else:
        Process2(k+1)

