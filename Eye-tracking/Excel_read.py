import xlrd

def read_file(file_url):
    try:
        data = xlrd.open_workbook(file_url)
        return data
    except Exception as e:
        print(str(e))

def filter_excel(workbook, column_name=0, by_name='Data'):
    '''
    :param workbook:
    :param column_name:
    :param by_name: 对应的sheet页
    :return:
    '''
    table = workbook.sheet_by_name(by_name)     #获得表格
    total_rows = table.nrows                    #拿到总行数
    columns = table.row_values(column_name)     #某一行的数据['姓名',‘用户名’，‘联系方式’]
    excel_list = []
    timestamp_list = []
    for one_row in range(1,total_rows):
        row = table.row_values(one_row)
        if row:
            row_object = {}
            for i in range(0,len(columns)):
                row_object[columns[i]] = row[i]
            excel_list.append(row_object)
            timestamp_list.append(row[6])
    return excel_list,timestamp_list

def find_index(time_List, Time):
    Time = round(Time*1000)
    ind = 0
    possible_time_list = [Time, Time-1, Time-2, Time-3, Time-4, Time-5, Time-6, Time-7, Time-8, \
                          Time-9, Time-10, Time-11, Time-12, Time-13, Time-14, Time-15, Time-16]
    for i in possible_time_list:
        if i in time_List:
            ind = time_List.index(i)
    # if (ind==0):
    #     print(Time)
    return ind

def valid_frames(excel_list, Sections):
    cnt = 0
    tot = 0
    flag = False
    for row_object in excel_list:
        for section in Sections:
            if row_object['RecordingTimestamp']/1000 in section:
                flag = True
                tot += 1
        if flag:
            if row_object['ValidityLeft']*row_object['ValidityRight']==0:
                cnt += 1
                # print(row_object['GazePointIndex'])
            flag = False
    return cnt