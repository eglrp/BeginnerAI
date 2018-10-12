import torch as t

class YoLoV1Loss(t.nn.Module):
    def __init__(self,CFG):
        super(YoLoV1Loss,self).__init__()
        self.CFG = CFG
        self.S = self.CFG["CELL_NUMS"]    #7代表将图像分为7x7的网格，将整个图片分为7*7的网格，图片尺寸为448*448，相当于一个格子的大小是64*64
        self.B = self.CFG["EACH_CELL_BOXES"]    #2代表一个网格预测两个框
        self.l_coord = self.CFG["L_COORD"]   #5代表 λcoord  更重视8维的坐标预测
        self.l_noobj = self.CFG["EACH_CELL_BOXES"]   #0.5代表没有object的bbox的confidence loss

    def compute_iou(self, box1, box2):

        N = box1.size(0)
        M = box2.size(0)

        lt = t.max(
            box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        rb = t.min(
            box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
            box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
        )

        wh = rb - lt  # [N,M,2]
        # wh(wh<0)= 0  # clip at 0
        wh= (wh < 0).float()
        inter = wh[:,:,0] * wh[:,:,1]  # [N,M]

        area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
        area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
        area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
        area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

        iou = inter / (area1 + area2 - inter)
        return iou

    '''
    pred_tensor : 预测值，[BatchSize,7,7,30]
    target_tensor : 目标值，[BatchSize,7,7,30]
    
    pred_tensor是经过神经网络计算出来的结果，我们的目标是要pred_tensor与target_tensor越来越接近，也就是损失越小，这就包含两个含义
    1. target_tensor中的方格里面需要进行物体预测的格子，在pred_tensor的响应格子上也应该是要进行物体预测的，并且物体预测的类别应该一样
    2. target_tensor中预测物体类别的格子所对应的box，在pred_tensor上的对应box也应该一样
    '''
    def forward(self,pred_tensor,target_tensor):

        # N为batchsize
        N = pred_tensor.size()[0] # N为batchSize

        '''
        我们知道，每个格子的维度都是30=2*5+10，前5位是第一个预测框的信息，接下来的5位是第二个预测框的信息，最后20位是当前格子预测类别的one-hot编码，我们当初
        在编码ground truth的时候，将需要负责预测类别的格子的4，9两个位置设置为1表明这个格子需要负责预测物体，也就是说有某个物体的中心点落在了这个格子里，所以
        我们根据target_tensor[:,:,:,4] > 0就可以找到那些负责预测物体的格子，target_tensor[:,:,:,4] == 0就可以找到不负责预测物体的格子。
        
        coo_mask 大部分为0  记录为1代表真实有物体的网格，0表示没有物体的格子
        noo_mask 大部分为1  记录为1代表真实无物体的网格，0表示有物体的格子。
        '''
        coo_mask = target_tensor[:,:,:,4] > 0
        noo_mask = target_tensor[:,:,:,4] == 0

        # coo_mask、noo_mask形状扩充到[32,7,7,30]
        coo_mask = coo_mask.unsqueeze(-1).expand_as(target_tensor)
        noo_mask = noo_mask.unsqueeze(-1).expand_as(target_tensor)

        '''
        coo_mask可以找到需要负责预测物体的格子，那么pred_tensor[coo_mask]，就相当于在预测结果中也找到对应的格子，这些格子也应该预测跟target一样的物体，如果不是，
        那么就产生了Loss，我们就需要负责减少这个loss
        '''
        # coo_pred 取出预测结果中有物体的网格，并改变形状为（xxx,30）  xxx代表一个batch的图片上的存在物体的网格总数    30代表2*5+20   例如：coo_pred[72,30]
        coo_pred = pred_tensor[coo_mask].view(-1,30)
        # 一个网格预测的两个box  30的前10即为2个x,y,w,h,c，并调整为（xxx,5） xxx为所有真实存在物体的预测框，而非所有真实存在物体的网格     例如：box_pred[144,5]
        # contiguous将不连续的数组调整为连续的数组
        box_pred = coo_pred[:,:10].contiguous().view(-1,5) #box[x1,y1,w1,h1,c1]
        # #[x2,y2,w2,h2,c2]
        # 每个网格预测的类别  后20
        class_pred = coo_pred[:,10:]

        # 对真实标签做同样操作
        coo_target = target_tensor[coo_mask].view(-1,30)
        box_target = coo_target[:,:10].contiguous().view(-1,5)
        class_target = coo_target[:,10:]

        # 计算不包含obj损失  即本来无，预测有
        # 在预测结果中拿到真实无物体的网格，并改变形状为（xxx,30）  xxx代表一个batch的图片上的不存在物体的网格总数    30代表2*5+20   例如：[1496,30]
        noo_pred = pred_tensor[noo_mask].view(-1,30)
        noo_target = target_tensor[noo_mask].view(-1,30)      # 例如：[1496,30]
        # ByteTensor：8-bit integer (unsigned)
        noo_pred_mask = t.cuda.ByteTensor(noo_pred.size()) if t.cuda.is_available() else t.ByteTensor(noo_pred.size())  # 例如：[1496,30]
        noo_pred_mask.zero_()   #初始化全为0
        # 将第4、9  即有物体的confidence置为1
        noo_pred_mask[:,4]=1;
        noo_pred_mask[:,9]=1
        # 拿到第4列和第9列里面的值（即拿到真实无物体的网格中，网络预测这些网格有物体的概率值）
        # 一行有两个值（第4和第9位）
        # 例如noo_pred_c：2992        noo_target_c：2992
        # noo pred只需要计算类别c的损失
        noo_pred_c = noo_pred[noo_pred_mask]
        # 拿到第4列和第9列里面的值  真值为0，真实无物体（即拿到真实无物体的网格中，这些网格有物体的概率值，为0）
        noo_target_c = noo_target[noo_pred_mask]
        # 均方误差    如果 size_average = True，返回 loss.mean()。    例如noo_pred_c：2992        noo_target_c：2992
        # nooobj_loss 一个标量
        nooobj_loss = t.nn.functional.mse_loss(noo_pred_c,noo_target_c,size_average=False)


        #计算包含obj损失  即本来有，预测有  和  本来有，预测无
        coo_response_mask = t.cuda.ByteTensor(box_target.size()) if t.cuda.is_available() else t.ByteTensor(box_target.size())
        coo_response_mask.zero_()
        coo_not_response_mask = t.cuda.ByteTensor(box_target.size()) if t.cuda.is_available() else t.ByteTensor(box_target.size())
        coo_not_response_mask.zero_()
        # 选择最好的IOU
        for i in range(0,box_target.size()[0],2):
            box1 = box_pred[i:i+2]
            box1_xyxy = t.autograd.Variable(t.FloatTensor(box1.size()).cuda() if t.cuda.is_available() else t.FloatTensor(box1.size()))
            box1_xyxy[:,:2] = box1[:,:2] -0.5*box1[:,2:4]
            box1_xyxy[:,2:4] = box1[:,:2] +0.5*box1[:,2:4]
            box2 = box_target[i].view(-1,5)
            box2_xyxy = t.autograd.Variable(t.FloatTensor(box2.size()).cuda() if t.cuda.is_available() else t.FloatTensor(box2.size()))
            box2_xyxy[:,:2] = box2[:,:2] -0.5*box2[:,2:4]
            box2_xyxy[:,2:4] = box2[:,:2] +0.5*box2[:,2:4]
            iou = self.compute_iou(box1_xyxy[:,:4],box2_xyxy[:,:4]) #[2,1]
            max_iou,max_index = iou.max(0)
            max_index = max_index.data.cuda() if t.cuda.is_available() else max_index.data
            coo_response_mask[i+max_index]=1
            coo_not_response_mask[i+1-max_index]=1
        # 1.response loss响应损失，即本来有，预测有   有相应 坐标预测的loss  （x,y,w开方，h开方）参考论文loss公式
        # box_pred [144,5]   coo_response_mask[144,5]   box_pred_response:[72,5]
        # 选择IOU最好的box来进行调整  负责检测出某物体
        box_pred_response = box_pred[coo_response_mask].view(-1,5)
        box_target_response = box_target[coo_response_mask].view(-1,5)
        # box_pred_response:[72,5]     计算预测 有物体的概率误差，返回一个数
        contain_loss = t.nn.functional.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        # 计算（x,y,w开方，h开方）参考论文loss公式
        loc_loss = t.nn.functional.mse_loss(box_pred_response[:,:2],box_target_response[:,:2],size_average=False) \
                   + t.nn.functional.mse_loss(t.sqrt(box_pred_response[:,2:4]),t.sqrt(box_target_response[:,2:4]),size_average=False)

        # 2.not response loss 未响应损失，即本来有，预测无   未响应
        # box_pred_not_response = box_pred[coo_not_response_mask].view(-1,5)
        # box_target_not_response = box_target[coo_not_response_mask].view(-1,5)
        # box_target_not_response[:,4]= 0
        # box_pred_response:[72,5]
        # 计算c  有物体的概率的loss
        not_contain_loss = t.nn.functional.mse_loss(box_pred_response[:,4],box_target_response[:,4],size_average=False)
        # 3.class loss  计算传入的真实有物体的网格  分类的类别损失
        class_loss = t.nn.functional.mse_loss(class_pred,class_target,size_average=False)
        # 除以N  即平均一张图的总损失
        return (self.l_coord*loc_loss + contain_loss + not_contain_loss + self.l_noobj*nooobj_loss + class_loss)/N