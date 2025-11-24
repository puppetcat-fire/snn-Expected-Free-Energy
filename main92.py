# import math
# from collections import defaultdict
threshold = 1.0
llr = 0
min_bw = 1e-4
# min_err = 1e-2
lr = 1e-5
def conn(in_node, out_node):
    out_node.set_in_node(in_node)
    in_node.set_out_node(out_node)



class Node:
    def __init__(self, id, o=1.02, b=-0.014):
        self.id = id
        self.o = o                  # 衰减/积累系数
        self.b = b

        self.cal_need = {
            "n":0,
            "oi^n":1,
            "sum(oi^n)":0,
            "sum((n-1)*oi^n)":0,
            "do":0,
            "db":0
        }
        self.increasing = False  # 单增标识
        self.start_cal = False
        self.in_conns = {}           # 输入连接
        self.out_nodes = []          # 输出连接
        self.cal_list = {}
        self.cal_finish = False       # 上次触发时间（ms）

        self.avgN = 400
        self.realN = 0
        self.hit_cnt = 0

        self.arouse_list = []
        self.arouse_line = -self.b/(self.o-1)

    def set_out_node(self, out_node):
        self.out_nodes.append(out_node)
    
    def set_in_node(self, in_node):
        self.in_conns[(in_node)] = {
            "w": 0.7,
            "t":[],
            "oi^t":[],
            "dw":0,
        }
        

    def arouse_next(self):
        for node in self.out_nodes:
            node.f_start_cal(self)
    
    def f_start_cal(self, in_node):
        if not self.cal_finish: # 如果已经计算完毕，则不重新激活
            self.start_cal = True
            # self.start_cal_N = True
            self.in_conns[(in_node)]["t"].append(0)
            self.in_conns[(in_node)]["oi^t"].append(1)

            self.calculated_monotonicity = False  # 计算过单调性

    def step(self, real_stats=False):
        """
        now_stats: 当前是否激活）
        """
        hit = False
        realN_cal = True
        if self.start_cal:
            self.realN+=1
            if self.realN > (self.avgN+1)*1.2:
                realN_cal = False
            if not self.cal_finish: # 没计算完
                energy = self.update_energy()
                self.arouse_list.append(energy)
                if not self.calculated_monotonicity:  # 每次前置节点激活后计算一次
                    if energy > -self.b/(self.o-1):
                        self.increasing = True
                        do, dw, db = self.cal_dall()
                        # 向-b/(o-1)*0.99拟合，或者说向-b/(o-1)的相反方向拟合，但问题是此刻，当下，o*St+wixi+b
                        # St = 0.99*(-b)/(o-1)
                        # Loss = (0.99*(-b)/(o-1) - St)^2
                        rate = 1-llr
                        # # if -self.b/(self.o-1) < -1 :
                        # #     dw = {node:0 for node in dw}
                        db += rate/(self.o-1)
                        do += -rate*self.b/(self.o-1)/(self.o-1)
                        # err = max(-self.b/(self.o-1)*rate-energy, -min_err)
                        err = -self.b/(self.o-1)*rate-energy
                        # err = 0.7-energy
                        self.cal_loss(err,do,dw,db,update_now=False)
                    else:
                        self.increasing = False
                        do, dw, db = self.cal_dall()
                        # 向-b/(o-1)*1.01拟合，或者说向-b/(o-1)的相反方向拟合，但问题是此刻，当下，o*St+wixi+b
                        # St = 1.01*(-b)/(o-1)
                        # St + 1.01*b/(o-1) = 0
                        rate = 1+llr
                        db += rate/(self.o-1)
                        do += -rate*self.b/(self.o-1)/(self.o-1)
                        err = -self.b/(self.o-1)*rate-energy
                        # err = min(-self.b/(self.o-1)*rate-energy, min_err)
                        # err = 0.7-energy
                        self.cal_loss(err,do,dw,db,update_now=False)
                    self.calculated_monotonicity = True
                # if self.realN<(self.avgN+1)*1.2:
                if energy>threshold:
                    if real_stats:  # 准时激活
                        hit = True
                    else:# 提前激活，将所有参与节点的值按比例减少
                        do, dw, db = self.cal_dall()
                        # S(t+1) = 1
                        # o*St+b = 1
                        # St = (1-b)/o
                        # St - (1-b)/o = 0
                        db += 1/self.o
                        do += (1-self.b)/self.o/self.o
                        # err = max((1-self.b)/self.o - energy, -min_err)
                        err = (1-self.b*self.o-self.b)/self.o/self.o - energy
                        self.cal_loss(err, do,dw,db, update_do=False)
                        # self.update_d()
                else:
                    if real_stats:
                        if not self.increasing:
                            self.update_d()
                        else:  #如果到达激活条件但是没能激活，
                            # St=o+b
                            # St - o - b = 0
                            do, dw, db = self.cal_dall()
                            do -= 1
                            db -= 1
                            # err = min(1+self.o+self.b-energy, min_err)
                            err = 1+self.o+self.b-energy
                            self.cal_loss(err,do,dw,db)
                    elif energy < 0:
                        self.reset_part()  # 用于更新状态后等待下次使用，但是保留do、db、和dwself.reset_part()  # 用于更新状态后等待下次使用，但是保留最近一次的do、db、和dw
                    #     else:
                    #         if real_stats:
                    #             if not self.increasing:  # 如果是递减的，则说明最后一次前置节点传入时未到达激活条件
                    #                 self.update_d()
                    #             
                #     else:
                #         # St=0
                #         # St = 0
                #         do, dw, db = self.cal_dall()
                #         # err = max(0 - energy, -min_err)
                #         err = 0-energy
                #         self.cal_loss(err, do,dw,db)
                #     self.reset()
        else:
            if real_stats:  # 如果没有激活，就说明最近一个小于0死掉了，这个时候do和dw还有db应该保留着最近的值
                self.update_d()
                realN_cal = False


        # 统计和激活后项是单独处理的
        if real_stats:
            if realN_cal:
                self.avgN = self.avgN*0.998 + 0.002*self.realN
                # if self.realN <250:
                #     breakpoint()
                # self.realN = 0
            # self.start_cal_N = False
            if self.id == "D":
                print(f"{self.hit_cnt:.16f}, {self.avgN:.2f}, {self.arouse_line:.24f}, {self.o:.24f} , {self.b:.24f}, {[self.in_conns[node]["w"] for node in self.in_conns]}")
            self.hit_cnt = self.hit_cnt*0.998
            if hit:
                self.hit_cnt += 0.002
            self.reset()
            self.arouse_next()

        # # 衰减/积累上一状态
        # hit = False

        # # # # 计算平均激活延迟
        # realN  = self.realN
        # if self.start_cal:
        #     if self.start_cal_N:
        #         self.realN += 1
        #         if real_stats:
        #             self.avgN = self.avgN*0.998 + 0.002*self.realN
        #             self.realN = 0
        #             self.start_cal_N = False
        
        # if self.start_cal: # 没开始计算  # 1111
        #     if not self.cal_finish: # 没计算完
        #         energy = self.update_energy()
        #         self.arouse_list.append(energy)
        #         if not self.calculated_monotonicity:  # 每次前置节点激活后计算一次
        #             if energy > -self.b/(self.o-1):
        #                 self.increasing = True
        #                 do, dw, db = self.cal_dall()
        #                 # 向-b/(o-1)*0.99拟合，或者说向-b/(o-1)的相反方向拟合，但问题是此刻，当下，o*St+wixi+b
        #                 # St = 0.99*(-b)/(o-1)
        #                 # Loss = (0.99*(-b)/(o-1) - St)^2

        #                 rate = 1-llr
        #                 # rate = 1
        #                 # if abs(do-0) > 1e-9:
        #                 if -self.b/(self.o-1) < -1 :
        #                     dw = {node:0 for node in dw}
        #                 db += rate/(self.o-1)
        #                 do += -rate*self.b/(self.o-1)/(self.o-1)
        #                 err = max(-self.b/(self.o-1)*rate-energy, -min_err)
        #                 self.cal_loss(err,do,dw,db,update_now=False)
        #             else:
        #                 self.increasing = False
        #                 do, dw, db = self.cal_dall()
        #                 # 向-b/(o-1)*1.01拟合，或者说向-b/(o-1)的相反方向拟合，但问题是此刻，当下，o*St+wixi+b
        #                 # St = 1.01*(-b)/(o-1)
        #                 # St + 1.01*b/(o-1) = 0
        #                 rate = 1+llr
        #                 # if abs(do-0)>1e-9:
        #                     # rate = 1
        #                 db += rate/(self.o-1)
        #                 do += -rate*self.b/(self.o-1)/(self.o-1)
        #                 err = min(-self.b/(self.o-1)*rate-energy, min_err)
        #                 self.cal_loss(err,do,dw,db,update_now=False)
        #             self.calculated_monotonicity = True
        #         if energy > threshold:  # 提前（或者准时）激活
        #             if real_stats:  # 准时激活
        #                 hit = True
        #                 self.reset()
        #             elif self.realN>self.avgN+1:
        #                 self.update_d()
        #                 self.reset()
        #             else:# 提前激活，将所有参与节点的值按比例减少
        #                 do, dw, db = self.cal_dall()
        #                 # S(t+1) = 1
        #                 # o*St+b = 1
        #                 # St = (1-b)/o
        #                 # St - (1-b)/o = 0
        #                 db += 1/self.o
        #                 do += (1-self.b)/self.o/self.o
        #                 err = max((1-self.b)/self.o - energy, -min_err)
        #                 self.cal_loss(err, do,dw,db)
        #                 self.reset_part()
        #         elif energy < 0:  # 小于零的直接舍弃
        #             # if real_stats:  # 如果此刻激活, energy<0，非增的话就不会来这
        #             #     if self.increasing:
        #             #         self.update_d()
        #             self.reset_part()
        #             if real_stats:
        #                 self.reset()
        #                 self.realN = 0
        #                 self.start_cal_N = False
        #         # elif self.realN>self.avgN*1.2:  # 超过处理平均值2倍的？要不T值超过两倍直接撇吗？弄一个大值然后快速衰减
        #         # # 。不用管静默，静默会因为-最后一次激活-激活值的递增，如果超远的w和o有值就带着呗，会在运动中平衡到一个固定值去
        #         #     # self.cal_abs(max((self.b-energy),-0.01),toZreo=True)
        #         #     self.realN = 0
        #         #     self.start_cal_N = False
        #         #     self.reset()
        #         else:  # 没到激活值值
        #             if real_stats:  # 如果此刻实际激活了
        #                 if not self.increasing:  # 如果是递减的，则说明最后一次前置节点传入时未到达激活条件
        #                     self.update_d()
        #                     self.reset()
        #                 else:  #如果到达激活条件但是没能激活，
        #                     # St=o+b
        #                     # St - o - b = 0
        #                     do, dw, db = self.cal_dall()
        #                     do -= 1
        #                     db -= 1
        #                     err = min(1+self.o+self.b-energy, min_err)
        #                     self.cal_loss(err,do,dw,db)
        #                 # self.cal_abs(min((self.o+self.b)-energy, 0.01), arouse_early=False)  # 未激活，将参与节点的值按比例增加
        #                 # self.cal(1.01-energy)  # 未激活，将参与节点的值按比例增加
        #                 self.realN = 0
        #                 self.start_cal_N = False
        #                 self.reset()
        #             elif self.realN>self.avgN*2:
        #                 if self.increasing:
        #                     self.update_d()
        #                     self.reset()
        #                 else:
        #                     # St=0
        #                     # St = 0
        #                     do, dw, db = self.cal_dall()
        #                     err = max(0 - energy, -min_err)
        #                     self.cal_loss(err, do,dw,db)
        #                     self.reset()
        #     else:
        #         if real_stats:
        #             self.realN = 0
        #             self.start_cal_N = False
        #             self.reset()
        #         # if self.realN>self.avgN*1.2:
        #         #     self.realN = 0
        #         #     self.start_cal_N = False
        # else:
        #     if real_stats:
        #         if not self.increasing:
        #             self.update_d()
        #         self.reset()
        #         # for conn in self.in_conns:
        #         #     if conn["w"] * 1.01 < 0.7:
        #         #         conn["w"] += max(0.01*abs(conn["w"]),0.0001)
        #         self.realN = 0
        #         self.start_cal_N = False
        # if real_stats:
        #     if self.id == "B":
        #         print(f"{self.hit_cnt:.16f}, {self.avgN:.2f}, {self.arouse_line:.24f}, {self.o:.24f} , {self.b:.24f}, {[str(self.in_conns[node]["w"])[:26] for node in self.in_conns]}")
        #     self.hit_cnt = self.hit_cnt*0.998
        #     if hit:
        #         self.hit_cnt += 0.002
        #     else:
        #         a = "c"
        #     self.realN = 0
        #     self.start_cal_N = False
        #     self.arouse_next()



    def reset(self):
        self.state = 0.0
        self.start_cal = False
        self.cal_finish = False
        for node in self.in_conns:
            self.in_conns[node]["t"] = []
            self.in_conns[node]["oi^t"] = []
            self.in_conns[node]["dw"] = 0
        self.cal_need = {
            "n":0,
            "oi^n":1,
            "sum(oi^n)":0,
            "sum((n-1)*oi^n)":0,
            "do":0,
            "db":0
        }
        self.realN = 0
        self.arouse_list = []

    def reset_part(self):
        self.state = 0.0
        self.start_cal = False
        # self.cal_finish = False
        for node in self.in_conns:
            self.in_conns[node]["t"] = []
            self.in_conns[node]["oi^t"] = []
        self.cal_need = {
            "n":0,
            "oi^n":1,
            "sum(oi^n)":0,
            "sum((n-1)*oi^n)":0,
            "do":self.cal_need["do"],
            "db":self.cal_need["db"]
        }
        self.realN = 0

    def update_energy(self):
        for node in self.in_conns:
            for i in range(len(self.in_conns[node]["t"])):
                self.in_conns[node]["t"][i] += 1
                if self.in_conns[node]["t"][i] > 1:
                    self.in_conns[node]["oi^t"][i] *= self.o
        if self.cal_need["n"] == 0:
            self.cal_need["sum((n-1)*oi^n)"] = 0
        elif self.cal_need["n"] == 1:
            self.cal_need["sum((n-1)*oi^n)"] = 1
        else:
            self.cal_need["sum((n-1)*oi^n)"] += self.cal_need["n"]*self.cal_need["oi^n"]
        self.cal_need["n"] += 1
        if self.cal_need["n"] > 1:
            self.cal_need["oi^n"] *= self.o
        self.cal_need["sum(oi^n)"] += self.cal_need["oi^n"]

        energy = self.b*self.cal_need["sum(oi^n)"]+sum([self.in_conns[node]["w"]*sum(self.in_conns[node]["oi^t"]) for node in self.in_conns])
        return energy

    def cal_dall(self):
        do = self.b*self.cal_need["sum((n-1)*oi^n)"]+sum([
                sum([
                    (self.in_conns[node]["t"][i]-1)*self.in_conns[node]["oi^t"][i]/self.o
                    for i in range(len(self.in_conns[node]["t"]))])*self.in_conns[node]["w"] 
                for node in self.in_conns])
        dw = {node: sum(self.in_conns[node]["oi^t"]) for node in self.in_conns}
        db = self.cal_need["sum(oi^n)"]
        return do, dw, db
    def cal_abs(self, err, do, dw, db, update_now=True):
        # if toZreo:
        #     do = self.b*self.cal_need["sum((n-1)*oi^n)"]+sum([
        #         sum([
        #             (conn["t"][i]-1)*conn["oi^t"][i]/self.o
        #             for i in range(len(conn["t"]))])*conn["w"] 
        #         for conn in self.in_conns])
        #     dw = [sum(conn["oi^t"]) for conn in self.in_conns]
        #     db = self.cal_need["sum(oi^n)"] - 1
        # else:
        #     do = self.b*self.cal_need["sum((n-1)*oi^n)"]+sum([
        #             sum([
        #                 (conn["t"][i]-1)*conn["oi^t"][i]/self.o
        #                 for i in range(len(conn["t"]))])*conn["w"] 
        #             for conn in self.in_conns])
        #     if arouse_early:
        #         do += -(1-self.b)/((self.o)*(self.o))
        #     else:
        #         do += 1
        #     dw = [sum(conn["oi^t"]) for conn in self.in_conns]
        #     db = self.cal_need["sum(oi^n)"]
        #     if arouse_early:
        #         db += -1/self.o
        #     else:
        #         db += 1
        # abs_sum = sum([abs(dw[i]*self.in_conns[i]["w"]) for i in range(len(self.in_conns))])+abs(db*self.b)
        abs_sum = abs(do*self.o)+sum([abs(dw[node]*self.in_conns[node]["w"]) for node in self.in_conns])+abs(db*self.b)
        
        # if self.b+ err*abs(db*self.b)/abs_sum/db < -2:
        #     a = 'c'

        # if (self.o + err*abs(do*self.o)/abs_sum/do + self.b + err*abs(db*self.b)/abs_sum/db)<1:
        #     a = "c"
        addo = 0
        if abs(do-0)>1e-9:
            addo = err*abs(do*self.o)/abs_sum/do
            if addo > 0.1:
                a = "c"
            if abs(db-0)>1e-9:
                addb = err*abs(db*self.b)/abs_sum/db
            else:
                addb = 0
            if self.o + addo <1:
                abs_sum -= abs(do*self.o) + abs(db*self.b)
                do = 0
                addo = 0
            elif self.o + addo > 1.5:
                abs_sum -= abs(do*self.o) + abs(db*self.b)
                do = 0
                addo = 0
            elif self.o+ addo + self.b + addb < 1:
                abs_sum -= abs(do*self.o) + abs(db*self.b)
                do = 0
                db = 0
                addo = 0
        if abs(db-0)>1e-9:
            if self.o+ addo + self.b + err*abs(db*self.b)/abs_sum/db < 1:
                abs_sum -= abs(db*self.b)
                db = 0

        if abs(do-0)>1e-9:
            addo = err*abs(do*self.o)/abs_sum/do
            if self.o + addo < 1:
                a = "c"
            if self.o+addo > 2:
                a = "c"
            if self.o+ addo + self.b + err*abs(db*self.b)/abs_sum/db < 1:
                a = "c"
        if update_now:
            if abs(do-0)>1e-9:
                if self.o + err*abs(do*self.o)/abs_sum/do > 2:
                    a = "c"
                self.o += err*abs(do*self.o)/abs_sum/do
                # if (self.o-0) < 1e-4:
                #     self.o *= -1
            if abs(db-0)>1e-9:
                self.b += err*abs(db*self.b)/abs_sum/db
            # self.oi = (0.7-self.b)/0.7
            
            for node in self.in_conns:
                if abs(dw[node]-0)>1e-9:
                    self.in_conns[node]["w"] += err*abs(dw[node]*self.in_conns[node]["w"])/abs_sum/dw[node]
                    if abs(self.in_conns[node]["w"]-0)<min_bw:
                        self.in_conns[node]["w"] *= -1
            self.arouse_line = -self.b/(self.o-1)
            self.cal_finish=True
        else:
            if abs(do-0)>1e-9:
                self.cal_need["do"] += err*abs(do*self.o)/abs_sum/do
                # if (self.o-0) < 1e-4:
                #     self.o *= -1
            if abs(db-0)>1e-9:
                self.cal_need["db"] += err*abs(db*self.b)/abs_sum/db
            # self.oi = (0.7-self.b)/0.7
            
            for node in self.in_conns:
                if abs(dw[node]-0)>1e-9:
                    self.in_conns[node]["dw"] += err*abs(dw[node]*self.in_conns[node]["w"])/abs_sum/dw[node]

    def cal_loss(self, err, do, dw, db, update_now=True, update_do = True):
        addo = lr*(2*err)*do
        addb = lr*(2*err)*db
        # if not update_do:
        #     addo = 0
        #     addb = 0
        addo = 0
        addb = 0
        if self.b+addb < -1:
            a = "c"
        addw = {node:lr*(2*err)*dw[node] for node in dw}
        if update_now:
            if (self.o+addo)>1.01 and (self.o+addo)<1.2:
                self.o += addo
            if (self.b+addb)<0 and (self.b+addb)>-0.1:
                self.b += addb
            for node in self.in_conns:
                    self.in_conns[node]["w"] += addw[node]
                    if abs(self.in_conns[node]["w"]-0)<min_bw:
                        self.in_conns[node]["w"] *= -1
            self.arouse_line = -self.b/(self.o-1)
            self.cal_finish=True
        else:
            # if (-self.b-addb)/(self.o+addo-1)<1:
            if (self.o+addo)>1.01 and (self.o+addo)<1.2:
                self.cal_need["do"] += addo
            if (self.b+addb)<0 and (self.b+addb)>-0.1:
                self.cal_need["db"] += addb
            
            for node in self.in_conns:
                self.in_conns[node]["dw"] += addw[node]

        
    
    def update_d(self):
        # self.o += self.cal_need["do"]

        # # self.cal_need["do"] = 0
        # self.b += self.cal_need["db"]
        # self.cal_need["db"] = 0
        for node in self.in_conns:
            self.in_conns[node]["w"] += self.in_conns[node]["dw"]
            self.in_conns[node]["dw"] = 0
            if abs(self.in_conns[node]["w"]-0)<min_bw:
                self.in_conns[node]["w"] *= -1
        self.arouse_line = -self.b/(self.o-1)
        
        


import random
random.seed(1)
if __name__=="__main__":
    # A = Node("A")
    # B = Node("B")
    # conn(A, B)

    # A.step()
    # B.step()

    # while True:
    #     A.step(True)
    #     # B.step()
    #     # B.step()
    #     # B.step()
    #     # B.step()
    #     B.step()
    #     B.step()
    #     B.step(True)
    #     input()
        
    #     # print("========================")


    # #     建立节点
    # A = Node("A")
    # B = Node("B")
    # C = Node("C")
    # conn(A, C)
    # conn(B, C)
    # while True:
    #     A.step(True)
    #     if random.random()>0.7:
    #         B.step(True)
    #     # B.step()
    #     # B.step()
    #     # B.step()
    #     # B.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step()
    #     C.step(True)
    #     # input()
    #     # print("========================")

    # # 建立节点
    A = Node("A")
    B = Node("B")
    C = Node("C")
    D = Node("D")

    conn(A, D)
    conn(B, D)
    conn(C, D)


    def mode1(A,B,C,D):
        for i in range(random.randint(400,1200)):
            A.step()
            B.step()
            C.step()
            D.step()
        A.step(True)
        B.step()
        C.step()
        D.step()
        for i in range(300):
            A.step()
            B.step()
            C.step()
            D.step()
        A.step()
        B.step()
        C.step()
        D.step(True)
    def mode2(A,B,C,D):
        for i in range(random.randint(400,1200)):
            A.step()
            B.step()
            C.step()
            D.step()

        A.step()
        B.step(True)
        C.step()
        D.step()
        for i in range(300):
            A.step()
            B.step()
            C.step()
            D.step()
        A.step()
        B.step()
        C.step()
        D.step(True)
    def mode3(A,B,C,D):
        for i in range(random.randint(400,1200)):
            A.step()
            B.step()
            C.step()
            D.step()
        A.step(True)
        B.step(True)
        C.step(True)
        D.step()
        for i in range(1800):
            A.step()
            B.step()
            C.step()
            D.step()
        A.step()
        B.step()
        C.step()
        D.step()
    while True:
        mode = random.randint(1,3)
        # print(mode)
        if mode==1:
            mode1(A,B,C,D)
        elif mode==2:
            mode2(A,B,C,D)
        elif mode==3:
            mode3(A,B,C,D)


    #   0.0000000000000000, 200, 1.0000001745026095 , -0.0031497929293083, ['0.3898380995369165', '0.3889942072969501', '0.2274671039725252']


    # # # # #  # 建立节点
    # A = Node("A")
    # B = Node("B")

    # conn(A, B)


    # def mode1(A,B):
    #     for i in range(random.randint(252,500)):
    #         A.step()
    #         B.step()
    #     A.step(True)
    #     B.step()
    #     for i in range(30):
    #         A.step()
    #         B.step()
    #     A.step(True)
    #     B.step()
    #     for i in range(30):
    #         A.step()
    #         B.step()
    #     A.step()
    #     B.step(True)
    # def mode2(A,B):
    #     for i in range(random.randint(252,500)):
    #         A.step()
    #         B.step()
    #     A.step()
    #     B.step()
    #     for i in range(30):
    #         A.step()
    #         B.step()
    #     A.step(True)
    #     B.step()
    #     for i in range(30):
    #         A.step()
    #         B.step()
    #     A.step()
    #     B.step()
    # def mode3(A,B):
    #     for i in range(random.randint(252,500)):
    #         A.step()
    #         B.step()
    #     A.step(True)
    #     B.step()
    #     for i in range(30):
    #         A.step()
    #         B.step()
    #     A.step()
    #     B.step()
    #     for i in range(30):
    #         A.step()
    #         B.step()
    #     A.step()
    #     B.step()
    # while True:
    #     mode = random.randint(1,3)
    #     # print(mode)
    #     if mode==1:
    #         mode1(A,B)
    #     elif mode==2:
    #         mode2(A,B)
    #     elif mode==3:
    #         mode3(A,B)
