from node import conn_node, Node,Conn
import random
import math
import itertools
import logging
T = 1
lr = 1e-2
# def conn_node(in_node, out_node):
#     out_node.set_in_node(in_node)
#     in_node.set_out_node(out_node)
class doConn(Conn):
    def __init__(self):
        self.tau_rise = 0.75
        self.tau_decay = 0.8
        self.A = -2.52
        self.freeezeList = {}  # 冻结列表。等待指定信号回传
        self.factor_rise = math.exp(-1/math.exp(self.tau_rise))
        self.factor_decay = math.exp(-1/math.exp(self.tau_decay))
        self.max_point = (self.tau_decay-self.tau_rise)/(math.exp(-self.tau_rise)-math.exp(-self.tau_decay))
        self.energy = []
    def new_cal(self):
        item = {
            "i": 0,
            "rising": math.exp(self.A),
            "decaying": math.exp(self.A),
            "i_list":[],
            "rising_list": [],
            "decaying_list": [],
        }
        self.energy.append(item)
        energy_list = []
        item['rising'] *= self.factor_rise
        item['decaying'] *= self.factor_decay
        item['i'] += 1
        item["i_list"].append(item['i'])
        item["rising_list"].append(item['rising'])
        item["decaying_list"].append(item['decaying'])
        target_erengy = item['decaying'] - item['rising']
        while target_erengy>1e-6:
            item['rising'] *= self.factor_rise
            item['decaying'] *= self.factor_decay
            item['i'] += 1
            item["i_list"].append(item['i'])
            item["rising_list"].append(item['rising'])
            item["decaying_list"].append(item['decaying'])
            target_erengy = item['decaying'] - item['rising']
            energy_list.append(target_erengy)
        return energy_list
    def step(self):
        for item in self.energy:
            del item['i_list'][0]
            del item['rising_list'][0]
            del item['decaying_list'][0]
    def freeze(self,id):
        for item in self.energy:
            self.freeezeList[id]={
                "rising":item["rising_list"][0],
                "decaying":item["decaying_list"][0],
                "i":item["i_list"][0],
                "tau_rise":self.tau_rise,
                "tau_decay": self.tau_decay,
                "A":self.A
            }
        self.reset()

    def update_params(self, id, energy, target_energy):
        loss = target_energy-energy
        rising = self.freeezeList[id]['rising']
        decaying = self.freeezeList[id]["decaying"]
        i = self.freeezeList[id]["i"]
        del self.freeezeList[id]
        dA = decaying - rising
        d_decay = decaying*i*math.exp(-self.tau_decay)
        d_rise = -rising*i*math.exp(-self.tau_rise)

        if self.tau_decay - lr*2*loss*d_decay < 1e-15:
            a = 'c'
        if self.tau_rise - lr*2*loss*d_rise < 1e-15:
            a = 'c'
        self.A -= lr*2*loss*dA
        self.tau_decay -= lr*2*loss*d_decay
        self.tau_rise -= lr*2*loss*d_rise
        self.factor_rise = math.exp(-1/math.exp(self.tau_rise))
        self.factor_decay = math.exp(-1/math.exp(self.tau_decay))
        self.max_point = (self.tau_decay-self.tau_rise)/(math.exp(-self.tau_rise)-math.exp(-self.tau_decay))
        
    def reset(self):
        self.energy = []
class doNode(Node):
    def __init__(self, id, show = False):
        self.id = id
        self.in_conns = {}           # 输入连接
        self.out_nodes = []          # 输出连接
        self.already_arouse = False
        self.next_step_reset = False
        self.arouse_time = 0
        self.hit_cnt = 0
        self.max_hit_cnt = 0
        self.start_cal_conn = []
        self.step_cnt = 0
        self.freeze_energy_List = {}
        self.enable_param_List = {}
        self.energylist =[]
        self.next_id = 0
        self.chosen_index = -1
        self.logger = None
        self._setup_logger()  # 初始化时设置logger

    def f_start_cal(self, in_node):
        # math.exp(target_erengy/T)
        conn = self.in_conns[(in_node)]
        energylist = conn.new_cal()

        self.energylist = [x + y for x, y in itertools.zip_longest(energylist, self.energylist, fillvalue=0)]
        sum_P = sum(self.energylist)
        for i in range(len(energylist)):
            energylist[i]/=sum_P
        # 根据概率随机选择序号（索引）
        indices = list(range(len(energylist)))  # 生成索引列表 [0, 1, 2, ...]
        self.chosen_index = random.choices(indices, weights=energylist, k=1)[0]  # k=1表示选择1个元素
        self.chosen_energy = energylist[self.chosen_index]
        self.step_cnt = 0

        # 现在可以使用 chosen_index 进行后续操作
    def step(self):
        self.step_cnt += 1
        barouse = False
        if self.step_cnt == self.chosen_index:
            barouse = True
            self.arouse_next()
            for in_node in self.in_conns:
                conn = self.in_conns[(in_node)]
                conn.freeze(self.next_id)
            self.freeze_energy_List[self.next_id] = self.energylist[0]
            self.enable_param_List[self.next_id] = {node:1 for node in self.out_nodes}
            self.next_id += 1
            self.reset()
        else:
            if self.energylist:
                del self.energylist[0]
        return barouse
    
    def reset(self):
        self.step_cnt = 0
        self.chosen_index = -1
        self.chosen_energy = -1
        self.energylist = []

    def arouse_next(self):
        for node in self.out_nodes:
            node.f_start_cal(self, id=self.next_id,returnParams=True)

    def update_param(self, node, id, real_energy):
        print_list = []
        for in_node in self.in_conns:
            print_list.append([math.exp(self.in_conns[(in_node)].A),math.exp(self.in_conns[(in_node)].tau_rise),math.exp(self.in_conns[(in_node)].tau_decay)])
            
        self.log(logging.DEBUG, f"{self.id}, {real_energy}, {print_list}")
        self.enable_param_List[id][node] = 0
        if sum([self.enable_param_List[id][item] for item in self.enable_param_List[id]]) == 0:
            for in_node in self.in_conns:
                conn = self.in_conns[(in_node)]
                conn.update_params(id, self.freeze_energy_List[id], real_energy)
            del self.freeze_energy_List[id]
            del self.enable_param_List[id]
    def remove_param(self, node, id, real_energy):
        print_list = []
        for in_node in self.in_conns:
            print_list.append([math.exp(self.in_conns[(in_node)].A),math.exp(self.in_conns[(in_node)].tau_rise),math.exp(self.in_conns[(in_node)].tau_decay)])
            
        self.log(logging.DEBUG, f"{self.id}, {real_energy}, {print_list}")
        self.enable_param_List[id][node] = 0
        if sum([self.enable_param_List[id][item] for item in self.enable_param_List[id]]) == 0:
            # for in_node in self.in_conns:
            #     conn = self.in_conns[(in_node)]
            #     conn.update_params(id, self.freeze_energy_List[id], real_energy)
            del self.freeze_energy_List[id]
            del self.enable_param_List[id]

    def set_in_node(self, in_node):
        self.in_conns[(in_node)] = doConn()

    def set_out_node(self, out_node):
        self.out_nodes.append(out_node)

def main():
    a = Node("A")
    b = doNode("B",show = True)
    c = Node("C",show=True)
    # conn_node(a,b)
    conn_node(a,c)
    conn_node(b,c)

    # while 1:
    #     for i in range(random.randint(50,100)):
    #         # b.step()
    #         c.step()
    #     a.arouse_next()
    #     barouse = False
    #     for i in range(20):
    #         c.step()
    #     if random.random()>0.5:
    #         b.arouse_next()
    #         barouse = True
    #     for i in range(20):
    #         c.step()
    #     if barouse:
    #         c.step(True)
    #         barouse = False
    #     else:
    #         c.step()
    while 1:
        barouse = False
        for i in range(random.randint(50,100)):
            # b.step()
            c.step()
        a.arouse_next()
        for i in range(20):
            # b.step()
            c.step()
        # if b.step():
        #     barouse = True
        if random.random()>0.5:
            b.arouse_next()
        for i in range(20):
            # b.step()
            c.step()
        if barouse:
            if random.random()>0.2:
                c.step(True)
            barouse = False
        else:
            if random.random()>0.8:
                c.step(True)

if __name__=="__main__":
    main()