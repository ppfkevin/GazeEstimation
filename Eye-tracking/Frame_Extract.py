import os
import cv2
from interval import Interval
from Collect_Infor import Collect_information1
from Excel_read import read_file,filter_excel,find_index,valid_frames

def Process(k):
    #6.089s:60
    valid_time = 8.357      #有效帧对应时间戳
    valid_frame = 132        #同步有效帧
    start_time = 0.743        #同步视频开始时间
    finish_time = 10.696    #同步视频结束时间
    synchronous_time = round((1 - valid_frame/600) * (finish_time-start_time) + valid_time, 3)    #同步视频播放结束对应时间戳

    discrete_start = - finish_time + synchronous_time + 16.497
    discrete_end = - finish_time + synchronous_time + 2*60 + 46.429
    section1 = Interval(discrete_start,discrete_end)
    continous_start = - finish_time + synchronous_time + 3*60 + 2.326
    continus_end = - finish_time + synchronous_time + 4*60 + 32.365
    section2 = Interval(continous_start,continus_end)
    Sections = [section1, section2]
    video_end = continus_end

    tobii_discrete_start = 16.497
    tobii_discrete_end = 2*60 + 46.429
    tobii_section1 = Interval(tobii_discrete_start, tobii_discrete_end)
    tobii_continous_start = 3*60 + 2.326
    tobii_continus_end = 4*60 + 32.365
    tobii_section2 = Interval(tobii_continous_start, tobii_continus_end)
    tobii_Sections = [tobii_section1, tobii_section2]

    root_path = 'F:\images\SJTUGaze\Pang_data\P16'

    gopro_paths = [os.path.join(root_path,'Eyetracking\GP1'), os.path.join(root_path,'Eyetracking\GP2'), os.path.join(root_path,'Eyetracking\GP3'), os.path.join(root_path,'Eyetracking\GP4')]
    gopro_path = gopro_paths[k]
    person = 'p016_'
    images_path = os.path.join(gopro_path,'Samples')
    file = read_file(os.path.join(root_path,'Zhengchuanyang_New test_wushun2_no_wushun2.xlsx'))
    tot_list, time_list = filter_excel(file)                       #excel表格中所有元素汇集
    videos_name = ['GOPR0575.MP4', 'GOPR0417.MP4', 'GOPR0274.MP4', 'GOPR0388.MP4']
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
        Process(k+1)

