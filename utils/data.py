class DetectData:
    def __init__(self, parts,labels):
        """ 自定义检测数据类型:
        name = 零件名称
        kind = 缺陷类别
        coordinate_x = x坐标
        coordinate_y = y坐标
        confidence = 置信度
        """
        self.label_dic = labels
        self.parts = parts
        self.name = None
        self.num = None
        self.kind = []
        self.coordinate_x = []
        self.coordinate_y = []
        self.confidence = []
    
    def add_properties(self, number, kind, confidence, coordinate_x, coordinate_y):
        self.clear_properties()
        self.num = number
        for i in range(number):
            if int(kind[i]) not in self.parts:
                self.kind.append(self.label_dic[int(kind[i])])
                self.coordinate_x.append(coordinate_x[i]) 
                self.coordinate_y.append(coordinate_y[i])
                self.confidence.append(confidence[i])
            else:
                self.name = self.label_dic[int(kind[i])]
                self.num = self.num-1
    
    def show_properties(self):
        print(f"物件名称--{self.name}")
        for i in range(self.num):
            print(f"{'##'}缺陷类型--{self.kind[i]}")
            print(f"{'##'*2}中心坐标X--{self.coordinate_x[i]}")
            print(f"{'##'*2}中心坐标Y--{self.coordinate_y[i]}")
            print(f"{'##'*2}置信度--{self.confidence[i]}")

    def clear_properties(self):
        self.name = None
        self.num = None
        self.kind = []
        self.coordinate_x = []
        self.coordinate_y = []
        self.confidence = []

class DetectData_v5:
    def __init__(self):
        """ 自定义检测数据类型
        name = 零件名称
        kind = 缺陷类别
        coordinate_x = x坐标
        coordinate_y = y坐标
        confidence = 置信度
        """
        self.name = "0"
        self.coordinate_x = 0
        self.coordinate_y = 0
        self.confidence = 0

    def show(self):
        print(f"物件名称--{self.name}")
        print(f"中心坐标X--{self.coordinate_x}")
        print(f"中心坐标Y--{self.coordinate_y}")
        print(f"置信度--{self.confidence}")

 