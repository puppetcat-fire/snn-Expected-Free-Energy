import math
import logging
import os
threshold = 1.0
gama = 1.0 # 高于最末值的loss权重
lr = 1e-2
def conn_node(in_node, out_node):
    out_node.set_in_node(in_node)
    in_node.set_out_node(out_node)


class Conn:
    def __init__(self):
        self.tau_rise = 0.75
        self.tau_decay = 0.8
        self.A = -4.82
        self.energy = []

        self.dA = 0
        self.dtau_rise = 0
        self.dtau_decay = 0

        self.exp_rise = math.exp(-self.tau_rise)
        self.exp_decay = math.exp(-self.tau_decay)
        self.factor_rise = math.exp(-1/math.exp(self.tau_rise))
        self.factor_decay = math.exp(-1/math.exp(self.tau_decay))

        # 调试用
        self.maxI = -1
        self.maxi = -1
    
    def new_cal(self):
        self.energy.append({
            "i": 0,
            "rising": math.exp(self.A),
            "decaying": math.exp(self.A),
            "i_list":[],
            "rising_list": [],
            "decaying_list": [],
            "sum_energy": []
        })

    def step(self):
        sum_energy = 0
        for item in self.energy:
            if item['i'] == 0:
                item['i'] += 1
            else:
                item['rising'] *= self.factor_rise
                item['decaying'] *= self.factor_decay
                item['i_list'].append(item['i'])
                item['rising_list'].append(item['rising'])
                item['decaying_list'].append(item['decaying'])
                sum_energy+= item['decaying'] - item['rising']
                item['i'] += 1
        return sum_energy

    def set_sum_energy(self, sum_energy):
        for item in self.energy:
            if item['i'] != 0:
                item['sum_energy'].append(sum_energy)
    
    def update_dparams(self, final_energy):
        self.dA = 0
        self.dtau_rise = 0
        self.dtau_decay = 0
        for item in self.energy:
            if item['i_list']:
                final_decaying = item['decaying_list'][-1]
                final_rising = item['rising_list'][-1]
                final_part_energy = final_decaying-final_rising
                final_i = item['i_list'][-1]-1
                for i in item['i_list']:
                    energy = item['sum_energy'][i-1]
                    decaying = item['decaying_list'][i-1]
                    rising = item['rising_list'][i-1]
                    part_energy = decaying - rising
                    if final_energy>0:
                        if energy > final_energy:
                            temp = 2*gama*(energy-final_energy)
                            self.dA += temp*(part_energy-final_part_energy)
                            self.dtau_decay += temp*self.exp_decay*(
                                (i-1)*decaying-final_i*final_decaying
                                )
                            self.dtau_rise -= temp*self.exp_rise*(
                                (i-1)*rising-final_i*final_rising
                                )
                    else:
                        if energy < final_energy:
                            temp = 2*gama*(energy-final_energy)
                            self.dA += temp*(part_energy-final_part_energy)
                            self.dtau_decay += temp*self.exp_decay*(
                                (i-1)*decaying-final_i*final_decaying
                                )
                            self.dtau_rise -= temp*self.exp_rise*(
                                (i-1)*rising-final_i*final_rising
                                )
                self.dA += 2*(final_energy-1)*(final_part_energy)
                self.dtau_decay += 2*(final_energy-1)*final_i*self.exp_decay*final_decaying
                self.dtau_rise -= 2*(final_energy-1)*final_i*self.exp_rise*final_rising
    
    def update_params(self):
        self.A -= lr*self.dA
        self.tau_rise -= lr*self.dtau_rise
        self.tau_decay -= lr*self.dtau_decay
        # print(self.A,self.tau_rise,self.tau_decay)
        self.exp_rise = math.exp(-self.tau_rise)
        self.exp_decay = math.exp(-self.tau_decay)
        self.factor_rise = math.exp(-1/math.exp(self.tau_rise))
        self.factor_decay = math.exp(-1/math.exp(self.tau_decay))
        

    def reset(self):
        self.energy = []

        
        

class Node:
    def __init__(self, id, show=False):
        self.id = id
        self.in_conns = {}           # 输入连接
        self.out_nodes = []          # 输出连接
        self.already_arouse = False
        self.next_step_reset = False
        self.arouse_time = 0
        self.hit_cnt = 0
        self.max_hit_cnt = 0
        self.start_cal_conn = []
        self.return_param_dict = []
        # test need
        self.MaxI = -10000
        self.MAxi = -10000
        self.i = 0
        self.logger = None
        self.log_cnt = 0
        self._setup_logger()  # 初始化时设置logger

    def _setup_logger(self):
        # """私有方法：为实例设置日志记录器"""
        # 创建logger实例，名称基于节点ID（确保唯一性）
        logger_name = f"node_{self.id}"
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)  # 设置日志级别（可根据需要调整）
        
        # 确保日志目录存在
        log_dir = "log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # 日志文件路径
        log_file = os.path.join(log_dir, f"{self.id}.log")
        
        # 创建FileHandler，指向日志文件
        handler = logging.FileHandler(log_file, encoding='utf-8')
        handler.setLevel(logging.DEBUG)
        
        # 设置日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # 避免重复添加handler（如果logger已有handler，则跳过）
        if not self.logger.handlers:
            self.logger.addHandler(handler)
    
    def log(self, level, message):
        """
        日志记录方法：所有日志操作通过此函数生成
        :param level: 日志级别（如 logging.INFO、logging.ERROR）
        :param message: 日志消息字符串
        """
        if self.logger is not None:
            self.logger.log(level, message)
        else:
            # 备用处理：如果logger未设置，打印错误（实际使用中应避免）
            print(f"Logger not setup for node {self.id}: {message}")
    def set_out_node(self, out_node):
        self.out_nodes.append(out_node)
    
    def set_in_node(self, in_node):
        self.in_conns[(in_node)] = Conn()
        
    def arouse_next(self):
        for node in self.out_nodes:
            node.f_start_cal(self)
    
    def f_start_cal(self, in_node, id=-1, returnParams=False):
        # if returnParams:
        #     self.return_param_dict.append([in_node, id])
        conn = self.in_conns[(in_node)]
        conn.new_cal()

    def get_sum_energy(self):
        sum_energy = 0
        this_energy_list = {}
        for in_node in self.in_conns:
            conn = self.in_conns[(in_node)]
            this_energy = conn.step()
            sum_energy += this_energy
            this_energy_list[in_node] = this_energy
        return sum_energy, this_energy_list
    def step(self, real_stats=False):
        """
        now_stats: 当前是否激活）
        """
        sum_energy, this_energy_list = self.get_sum_energy()
        for in_node in self.in_conns:
            conn = self.in_conns[(in_node)]
            conn.set_sum_energy(sum_energy)


        # test need
        self.i += 1
        if sum_energy > self.MaxI:
            self.MaxI = sum_energy
            self.MAxi = self.i
            self.Max_energy_list = this_energy_list
        # test need end
        if self.already_arouse:
            self.arouse_time += 1
        if sum_energy>threshold:
            # T-1激活，T激活，T+1激活都视为命中。命中后arouse——time自加一
            if not self.already_arouse:
                self.this_energy_list = this_energy_list
                self.already_arouse = True
        if self.next_step_reset:
            print_list = []
            for in_node in self.in_conns:
                print_list.append([math.exp(self.in_conns[(in_node)].A),math.exp(self.in_conns[(in_node)].tau_rise),math.exp(self.in_conns[(in_node)].tau_decay)])
            self.log_cnt += 1
            if self.log_cnt % 1000==0:
                self.log(logging.DEBUG, f"{self.id}, {self.arouse_time}, {self.i}, {self.hit_cnt}, {self.max_hit_cnt}, {self.MAxi}, {self.MaxI}")
                self.log(logging.DEBUG, f"{print_list}")
            # input()
            self.hit_cnt *= 0.998
            if self.arouse_time > 3 or not self.already_arouse:
                # 未命中，更新参数,参数没轮计算时都会重置成0,此处命中不用清理缓存
                for in_node in self.in_conns:
                    conn = self.in_conns[(in_node)]
                    conn.update_params()
                self.already_arouse = False
                self.arouse_time = 0
                self.next_step_reset = False
            else:
                self.hit_cnt += 0.002
                if self.hit_cnt>self.max_hit_cnt:
                    self.max_hit_cnt = self.hit_cnt
                self.already_arouse = False
                self.arouse_time = 0
                self.next_step_reset = False
            
            for in_node in self.in_conns:
                conn = self.in_conns[(in_node)]
                conn.reset()
            if self.already_arouse:
                this_energy_list = self.this_energy_list
            else:
                this_energy_list = self.Max_energy_list
            for node,id in self.return_param_dict:
                if node in this_energy_list:
                    energy = this_energy_list[node]
                    # if (energy -0)>1e-10:
                    #     if self.hit_cnt>0.2:
                    #         node.update_param(self, id, energy)
                    #     else:
                    #         node.remove_param(self, id, energy)
            self.return_param_dict = []
            self.this_energy_list = []
            self.MaxI = -10000
            self.MAxi = -10000
            self.i = 0
            
        if real_stats:
            self.next_step_reset = True
            if self.MAxi==1:
                a = "c"
            for in_node in self.in_conns:
                conn = self.in_conns[(in_node)]
                conn.update_dparams(sum_energy)

                

import random
random.seed(1)
if __name__=="__main__":
    A = Node("A")
    B = Node("B")
    conn_node(A, B)
    conn_node(B, B)

    A.step()
    B.step()

    while True:
        A.arouse_next()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        B.step()
        if random.random()>0.2:
            B.step(True)
        else:
            B.step()
        B.step()
        # input()
        
    #     # print("========================")


    # # # #     建立节点
    # A = Node("A")
    # B = Node("B")
    # C = Node("C")
    # conn_node(A, C)
    # conn_node(B, C)
    # while True:
    #     A.arouse_next()
    #     if random.random()>0.7:
    #         B.arouse_next()
    #     # B.step()
    #     # B.step()
    #     # B.step()
    #     # B.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step(True)
    #     C.step()
    #     # input()
    #     # print("========================")

    # # # 建立节点
    # A = Node("A")
    # B = Node("B")
    # C = Node("C")
    # D = Node("D")

    # conn_node(A, D)
    # conn_node(B, D)
    # conn_node(C, D)


    # def mode1(A,B,C,D):
    #     for i in range(random.randint(40,120)):
    #         D.step()
    #     A.arouse_next()
    #     D.step()
    #     for i in range(30):
    #         D.step()
    #     D.step(True)
    # def mode2(A,B,C,D):
    #     for i in range(random.randint(40,120)):
    #         D.step()
    #     B.arouse_next()
    #     D.step()
    #     for i in range(30):
    #         D.step()
    #     D.step(True)
    # def mode3(A,B,C,D):
    #     for i in range(random.randint(40,120)):
    #         D.step()
    #     A.arouse_next()
    #     B.arouse_next()
    #     C.arouse_next()
    #     D.step()
    #     for i in range(180):
    #         D.step()
    #     D.step()
    # while True:
    #     mode = random.randint(1,3)
    #     # print(mode)
    #     if mode==1:
    #         mode1(A,B,C,D)
    #     elif mode==2:
    #         mode2(A,B,C,D)
    #     elif mode==3:
    #         mode3(A,B,C,D)


    #   0.0000000000000000, 200, 1.0000001745026095 , -0.0031497929293083, ['0.3898380995369165', '0.3889942072969501', '0.2274671039725252']


    # # # # #  # 建立节点
    A = Node("A")
    B = Node("B",show=True)

    conn_node(A, B)


    def mode1(A,B):
        for i in range(random.randint(10,150)):
            B.step()
        A.arouse_next()
        B.step()
        for i in range(30):
            B.step()
        A.arouse_next()
        B.step()
        for i in range(30):
            B.step()
        B.step(True)
        B.step()
    def mode2(A,B):
        for i in range(random.randint(10,150)):
            B.step()
        B.step()
        for i in range(30):
            B.step()
        A.arouse_next()
        B.step()
        for i in range(30):
            B.step()
        B.step()
        B.step()
    def mode3(A,B):
        for i in range(random.randint(10,150)):
            B.step()
        A.arouse_next()
        B.step()
        for i in range(30):
            B.step()
        B.step()
        for i in range(30):
            B.step()
        B.step()
        B.step()
    while True:
        mode = random.randint(1,3)
        # print(mode)
        if mode==1:
            mode1(A,B)
        elif mode==2:
            mode2(A,B)
        elif mode==3:
            mode3(A,B)
