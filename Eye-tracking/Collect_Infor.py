def Collect_information1(indeX, Tot_list, W):
    tmp = Tot_list[indeX]
    if int(tmp['FixationIndex']) > 0:
        W.write('1 ')
    elif int(tmp['SaccadeIndex']) > 0:
        W.write('0 ')
    if tmp['ValidityLeft'] == 4:
        tmp['ValidityLeft'] = 0
        tmp['GazePointLeftX (ADCSpx)'] = 0
        tmp['GazePointLeftY (ADCSpx)'] = 0
        tmp['GazePointX (ADCSpx)'] = tmp['GazePointRightX (ADCSpx)']
        tmp['GazePointY (ADCSpx)'] = tmp['GazePointRightY (ADCSpx)']
        tmp['GazePointLeftX (ADCSmm)'] = 0
        tmp['GazePointLeftY (ADCSmm)'] = 0
        tmp['EyePosLeftX (ADCSmm)'] = 0
        tmp['EyePosLeftY (ADCSmm)'] = 0
        tmp['EyePosLeftZ (ADCSmm)'] = 0
    elif tmp['ValidityRight'] == 4:
        tmp['ValidityRight'] = 0
        tmp['GazePointRightX (ADCSpx)'] = 0
        tmp['GazePointRightY (ADCSpx)'] = 0
        tmp['GazePointX (ADCSpx)'] = tmp['GazePointLeftX (ADCSpx)']
        tmp['GazePointY (ADCSpx)'] = tmp['GazePointLeftY (ADCSpx)']
        tmp['GazePointRightX (ADCSmm)'] = 0
        tmp['GazePointRightY (ADCSmm)'] = 0
        tmp['EyePosRightX (ADCSmm)'] = 0
        tmp['EyePosRightY (ADCSmm)'] = 0
        tmp['EyePosRightZ (ADCSmm)'] = 0
    if (tmp['ValidityRight']==0) and (tmp['ValidityLeft']) == 0:
        W.write(str(tmp['GazePointLeftX (ADCSpx)']) + ' ')
        W.write(str(tmp['GazePointLeftY (ADCSpx)']) + ' ')
        W.write(str(tmp['GazePointRightX (ADCSpx)']) + ' ')
        W.write(str(tmp['GazePointRightY (ADCSpx)']) + ' ')
        W.write(str(tmp['GazePointX (ADCSpx)']) + ' ')
        W.write(str(tmp['GazePointY (ADCSpx)']) + ' ')
        W.write(str(tmp['GazePointLeftX (ADCSmm)']) + ' ')  #-10
        W.write(str(tmp['GazePointLeftY (ADCSmm)']) + ' ')
        W.write(str(tmp['GazePointRightX (ADCSmm)']) + ' ')
        W.write(str(tmp['GazePointRightY (ADCSmm)']) + ' ') #-7
        W.write(str(tmp['EyePosLeftX (ADCSmm)']) + ' ')
        W.write(str(tmp['EyePosLeftY (ADCSmm)']) + ' ')
        W.write(str(tmp['EyePosLeftZ (ADCSmm)']) + ' ')
        W.write(str(tmp['EyePosRightX (ADCSmm)']) + ' ')
        W.write(str(tmp['EyePosRightY (ADCSmm)']) + ' ')
        W.write(str(tmp['EyePosRightZ (ADCSmm)']) + ' ')    #-1
        W.write(str('\n'))
        return None

def Collect_information2(indeX, Tot_list, W):
    tmp = Tot_list[indeX]
    if tmp['ValidityLeft'] == 4:
        tmp['ValidityLeft'] = 0
        tmp['GazePointLeftX (ADCSpx)'] = 0
        tmp['GazePointLeftY (ADCSpx)'] = 0
        tmp['GazePointX (ADCSpx)'] = tmp['GazePointRightX (ADCSpx)']

        tmp['GazePointY (ADCSpx)'] = tmp['GazePointRightY (ADCSpx)']
    elif tmp['ValidityRight'] == 4:
        tmp['ValidityRight'] = 0
        tmp['GazePointRightX (ADCSpx)'] = 0
        tmp['GazePointRightY (ADCSpx)'] = 0
        tmp['GazePointX (ADCSpx)'] = tmp['GazePointLeftX (ADCSpx)']
        tmp['GazePointY (ADCSpx)'] = tmp['GazePointLeftY (ADCSpx)']

    if (tmp['ValidityRight'] == 0) and (tmp['ValidityLeft']) == 0:
        W.write(str(tmp['GazePointLeftX (ADCSpx)']) + ' ')
        W.write(str(tmp['GazePointLeftY (ADCSpx)']) + ' ')
        W.write(str(tmp['GazePointRightX (ADCSpx)']) + ' ')
        W.write(str(tmp['GazePointRightY (ADCSpx)']) + ' ')
        W.write(str(tmp['GazePointX (ADCSpx)']) + ' ')
        W.write(str(tmp['GazePointY (ADCSpx)']) + ' ')
        Gaze_vector_left = [tmp['GazePointLeftX (ADCSmm)']-tmp['EyePosLeftX (ADCSmm)'], \
                            tmp['GazePointLeftY (ADCSmm)']-tmp['EyePosLeftY (ADCSmm)'], \
                            0-tmp['EyePosLeftZ (ADCSmm)']]
        Gaze_vector_right = [tmp['GazePointRightX (ADCSmm)'] - tmp['EyePosRightX (ADCSmm)'], \
                            tmp['GazePointRightY (ADCSmm)'] - tmp['EyePosRightY (ADCSmm)'], \
                            0 - tmp['EyePosRightZ (ADCSmm)']]
        W.write(str(Gaze_vector_left[0])+' '+str(Gaze_vector_left[1])+' '+str(Gaze_vector_left[2])+' ')
        W.write(str(Gaze_vector_right[0]) + ' ' + str(Gaze_vector_right[1]) + ' ' + str(Gaze_vector_right[2]) + ' ')
        W.write(str('\n'))
        return None