import numpy as np
import torch
from utils.utils import *
import os
from dataset_modules.dataset_generic import save_splits
from models.model_mil import MIL_fc, MIL_fc_mc
from models.model_clam import CLAM_MB, CLAM_SB
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
import torch.nn as nn

# 设置设备,优先使用GPU,否则使用CPU
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Accuracy_Logger(object):
    """准确率记录器"""
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes # 类别数
        self.initialize()

    def initialize(self):
        # 为每个类别初始化统计数据
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        """记录单个样本的预测结果
        Args:
            Y_hat: 预测标签
            Y: 真实标签
        """
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1 # 该类别样本总数+1
        self.data[Y]["correct"] += (Y_hat == Y) # 如果预测正确,该类别正确数+1
    
    def log_batch(self, Y_hat, Y):
        """记录一个batch的预测结果
        Args:
            Y_hat: 预测标签数组
            Y: 真实标签数组
        """
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class # 获取当前类别的mask
            self.data[label_class]["count"] += cls_mask.sum() # 累加该类别的样本数
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum() # 累加预测正确的样本数
    
    def get_summary(self, c):
        """获取某个类别的统计结果
        Args:
            c: 类别索引
        Returns:
            acc: 准确率
            correct: 正确预测数
            count: 样本总数
        """
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None # 如果样本数为0,准确率返回None
        else:
            acc = float(correct) / count # 计算准确率
        
        return acc, correct, count

class EarlyStopping:
    """早停机制,当验证集损失在一定epoch内没有改善时停止训练"""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): 等待验证损失改善的最大epoch数,默认20
            stop_epoch (int): 最早可以停止的epoch
            verbose (bool): 是否打印详细信息
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0 # 计数器,记录验证损失未改善的epoch数
        self.best_score = None # 最佳得分
        self.early_stop = False # 是否早停的标志
        self.val_loss_min = np.inf # 最小验证损失

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):
        """
        Args:
            epoch: 当前epoch
            val_loss: 验证集损失
            model: 模型
            ckpt_name: 检查点文件名
        """
        score = -val_loss # 将损失转换为分数(损失越小分数越高)

        if self.best_score is None: # 首次调用
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score: # 分数未超过最佳分数
            self.counter += 1 # 计数器+1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch: # 如果超过耐心值且达到最小epoch
                self.early_stop = True # 设置早停标志
        else: # 分数超过最佳分数
            self.best_score = score # 更新最佳分数
            self.save_checkpoint(val_loss, model, ckpt_name) # 保存检查点
            self.counter = 0 # 重置计数器

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''当验证损失降低时保存模型'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name) # 保存模型状态
        self.val_loss_min = val_loss # 更新最小验证损失

def train(datasets, cur, args):
    """   
    训练单个fold
    Args:
        datasets: 包含训练集、验证集、测试集的数据集元组
        cur: 当前fold的索引
        args: 参数配置
    Returns:
        results_dict: 测试结果字典
        test_auc: 测试集AUC
        val_auc: 验证集AUC
        1-test_error: 测试集准确率
        1-val_error: 验证集准确率
    """
    print('\n' + '='*50)
    print(f'开始训练 Fold {cur + 1}/{args.k}')
    print('='*50)

    # 确保正确解包datasets
    train_split, val_split, test_split = datasets  # 确保这行代码在使用train_split之前

    print('\n数据集信息:')
    print(f'训练集: {len(train_split)} 样本')
    print(f'验证集: {len(val_split)} 样本') 
    print(f'测试集: {len(test_split)} 样本')

    print('\n模型配置:')
    print(f'模型类型: {args.model_type}')
    print(f'损失函数: {args.bag_loss}')
    print(f'优化器: {args.opt}')
    print(f'早停: {"启用" if args.early_stopping else "禁用"}')

    # 创建tensorboard日志目录
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)
    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    # 计算类别权重
    class_counts = [22, 51]  # 假设类别0有22个样本，类别1有51个样本
    total_count = sum(class_counts)
    weights = [total_count / count for count in class_counts]  # 计算权重
    weights = torch.tensor(weights).float().to(device)  # 转换为张量并移动到设备

    # 初始化损失函数
    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'svm':
        from topk.svm import SmoothTop1SVM
        loss_fn = SmoothTop1SVM(n_classes = args.n_classes)
        if device.type == 'cuda':
            loss_fn = loss_fn.cuda()
    elif args.bag_loss == 'weighted_ce':
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)  # 使用带权重的BCE损失
    else:
        loss_fn = nn.CrossEntropyLoss()
    print('Done!')
    
    # 初始化模型
    print('\nInit Model...', end=' ')
    model_dict = {"dropout": args.drop_out, 
                  'n_classes': args.n_classes, 
                  "embed_dim": args.embed_dim}
    
    if args.model_size is not None and args.model_type != 'mil':
        model_dict.update({"size_arg": args.model_size})
    
    if args.model_type in ['clam_sb', 'clam_mb']:
        if args.subtyping:
            model_dict.update({'subtyping': True})
        
        if args.B > 0:
            model_dict.update({'k_sample': args.B})
        
        if args.inst_loss == 'svm':
            from topk.svm import SmoothTop1SVM
            instance_loss_fn = SmoothTop1SVM(n_classes = 2)
            if device.type == 'cuda':
                instance_loss_fn = instance_loss_fn.cuda()
        elif args.inst_loss == 'weighted_ce':
            instance_loss_fn = nn.BCEWithLogitsLoss(pos_weight=weights)  # 使用带权重的BCE损失
        else:
            instance_loss_fn = nn.CrossEntropyLoss()
        
        if args.model_type =='clam_sb':
            model = CLAM_SB(**model_dict, instance_loss_fn=instance_loss_fn)
        elif args.model_type == 'clam_mb':
            model = CLAM_MB(**model_dict, instance_loss_fn=instance_loss_fn)
        else:
            raise NotImplementedError
    
    else: # args.model_type == 'mil'
        if args.n_classes > 2:
            model = MIL_fc_mc(**model_dict)
        else:
            model = MIL_fc(**model_dict)
    
    _ = model.to(device)
    print('Done!')
    print_network(model)

    # 初始化优化器
    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    # 初始化数据加载器
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, testing = args.testing, weighted = args.weighted_sample)
    val_loader = get_split_loader(val_split,  testing = args.testing)
    test_loader = get_split_loader(test_split, testing = args.testing)
    print('Done!')

    # 设置早停
    print('\nSetup EarlyStopping...', end=' ')
    early_stopping = EarlyStopping(patience = 10, stop_epoch=40, verbose = True) if args.early_stopping else None
    print('Done!')

    # 开始训练循环
    for epoch in range(args.max_epochs):
        print(f'\n【Epoch {epoch + 1}/{args.max_epochs}】')
        
        if args.model_type in ['clam_sb', 'clam_mb'] and not args.no_inst_cluster:     
            train_loss, train_error = train_loop_clam(epoch, model, train_loader, optimizer, args.n_classes, args.bag_weight, writer, loss_fn)
            val_loss, val_error, val_auc = validate_clam(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        else:
            train_loss, train_error = train_loop(epoch, model, train_loader, optimizer, args.n_classes, writer, loss_fn)
            val_loss, val_error, val_auc = validate(cur, epoch, model, val_loader, args.n_classes, 
                early_stopping, writer, loss_fn, args.results_dir)
        
        print(f'训练损失: {train_loss:.4f}, 训练错误率: {train_error:.4f}')
        print(f'验证损失: {val_loss:.4f}, 验证错误率: {val_error:.4f}, 验证AUC: {val_auc:.4f}')

        if early_stopping and early_stopping.early_stop:
            print("触发早停，训练结束")
            break

    # 加载最佳模型或保存最终模型
    if args.early_stopping:
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur))))
    else:
        torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))

    # 在测试集上评估
    results_dict, test_error, test_auc, acc_logger = summary(model, test_loader, args.n_classes)
    print('测试集错误率: {:.4f}, 测试集AUC: {:.4f}'.format(test_error, test_auc))

    # 打印每个类别的准确率
    for i in range(args.n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('类别 {}: 准确率 {}, 正确预测 {}/{}'.format(i, acc, correct, count))

        if writer:
            writer.add_scalar('final/test_class_{}_acc'.format(i), acc, 0)

    # 记录最终结果到tensorboard
    if writer:
        writer.add_scalar('final/val_error', val_error, 0)
        writer.add_scalar('final/val_auc', val_auc, 0)
        writer.add_scalar('final/test_error', test_error, 0)
        writer.add_scalar('final/test_auc', test_auc, 0)
        writer.close()

    # 打印最终结果
    print('\n' + '='*50)
    print('训练完成,最终结果:')
    print(f'验证集 - 错误率: {val_error:.4f}, AUC: {val_auc:.4f}')
    print(f'测试集 - 错误率: {test_error:.4f}, AUC: {test_auc:.4f}')
    print('='*50)

    return results_dict, test_auc, val_auc, 1-test_error, 1-val_error 


def train_loop_clam(epoch, model, loader, optimizer, n_classes, bag_weight, writer = None, loss_fn = None):
    """CLAM模型的训练循环
    Args:
        epoch: 当前epoch
        model: 模型
        loader: 数据加载器
        optimizer: 优化器
        n_classes: 类别数
        bag_weight: 包级别损失的权重
        writer: tensorboard写入器
        loss_fn: 损失函数
    """
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    
    train_loss = 0.
    train_error = 0.
    train_inst_loss = 0.
    inst_count = 0

    print('\n')
    for batch_idx, data in enumerate(loader):
        # 检查是否包含文本描述
        if len(data) == 3:  # 如果包含文本描述
            features, label, text_desc = data
            features, label = features.to(device), label.to(device)
            
            # 前向传播（传入文本描述）
            logits, Y_prob, Y_hat, _, instance_dict = model(features, label=label, instance_eval=True, text_descriptions=text_desc)
        else:  # 如果不包含文本描述
            features, label = data
            features, label = features.to(device), label.to(device)
            
            # 前向传播（不传入文本描述）
            logits, Y_prob, Y_hat, _, instance_dict = model(features, label=label, instance_eval=True)
        
        # 记录准确率
        acc_logger.log(Y_hat, label)
        
        # 计算包级别损失
        loss = loss_fn(logits, label)
        loss_value = loss.item()

        # 计算实例级别损失
        instance_loss = instance_dict['instance_loss']
        inst_count+=1
        instance_loss_value = instance_loss.item()
        train_inst_loss += instance_loss_value
        
        # 计算总损失
        total_loss = bag_weight * loss + (1-bag_weight) * instance_loss 

        # 记录实例级别预测
        inst_preds = instance_dict['inst_preds']
        inst_labels = instance_dict['inst_labels']
        inst_logger.log_batch(inst_preds, inst_labels)

        train_loss += loss_value
        
        # 每20个batch打印一次信息
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, instance_loss: {:.4f}, weighted_loss: {:.4f}, '.format(batch_idx, loss_value, instance_loss_value, total_loss.item()) + 
                'label: {}, bag_size: {}'.format(label.item(), features.size(0)))

        error = calculate_error(Y_hat, label)
        train_error += error
        
        # 反向传播
        total_loss.backward()
        # 优化器步进
        optimizer.step()
        optimizer.zero_grad()

    # 计算epoch的平均损失和错误率
    train_loss /= len(loader)
    train_error /= len(loader)
    
    # 打印实例级别的聚类准确率
    if inst_count > 0:
        train_inst_loss /= inst_count
        print('\n')
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))

    # 打印epoch总结
    print('Epoch: {}, train_loss: {:.4f}, train_clustering_loss:  {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_inst_loss,  train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer and acc is not None:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    # 记录训练指标到tensorboard
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)
        writer.add_scalar('train/clustering_loss', train_inst_loss, epoch)

    return train_loss, train_error

def train_loop(epoch, model, loader, optimizer, n_classes, writer = None, loss_fn = None):   
    """标准MIL模型的训练循环
    Args:
        epoch: 当前epoch
        model: 模型
        loader: 数据加载器
        optimizer: 优化器
        n_classes: 类别数
        writer: tensorboard写入器
        loss_fn: 损失函数
    """
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_loss = 0.
    train_error = 0.

    print('\n')
    for batch_idx, data in enumerate(loader):
        # 检查是否包含文本描述
        if len(data) == 3:  # 如果包含文本描述
            features, label, text_desc = data
            features, label = features.to(device, non_blocking=True), label.to(device, non_blocking=True)
            
            # 前向传播（传入文本描述）
            logits, Y_prob, Y_hat, _, _ = model(features, text_descriptions=text_desc)
        else:  # 如果不包含文本描述
        
            features, label = data
            features, label = features.to(device, non_blocking=True), label.to(device, non_blocking=True)
            
            # 前向传播（不传入文本描述）
            logits, Y_prob, Y_hat, _, _ = model(features)
        
        # 记录准确率
        acc_logger.log(Y_hat, label)
        
        # 计算损失
        loss = loss_fn(logits, label)
        loss_value = loss.item()
        
        train_loss += loss_value
        
        # 每20个batch打印一次信息
        if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss_value, label.item(), features.size(0)))
           
        error = calculate_error(Y_hat, label)
        train_error += error
        
        # 反向传播
        loss.backward()
        # 优化器步进
        optimizer.step()
        optimizer.zero_grad()

    # 计算epoch的平均损失和错误率
    train_loss /= len(loader)
    train_error /= len(loader)

    # 打印epoch总结
    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        if writer:
            writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    # 记录训练指标到tensorboard
    if writer:
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('train/error', train_error, epoch)

   
def validate(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir=None):
    """标准MIL模型的验证
    Args:
        cur: 当前fold索引
        epoch: 当前epoch
        model: 模型
        loader: 数据加载器
        n_classes: 类别数
        early_stopping: 早停对象
        writer: tensorboard写入器
        loss_fn: 损失函数
        results_dir: 结果保存目录
    Returns:
        bool: 是否触发早停
    """
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            # 检查是否包含文本描述
            if len(data) == 3:  # 如果包含文本描述
                features, label, text_desc = data
                features, label = features.to(device, non_blocking=True), label.to(device, non_blocking=True)
                
                # 前向传播（传入文本描述）
                logits, Y_prob, Y_hat, _, _ = model(features, text_descriptions=text_desc)
            else:  # 如果不包含文本描述
                features, label = data
                features, label = features.to(device, non_blocking=True), label.to(device, non_blocking=True)
                
                # 前向传播（不传入文本描述）
                logits, Y_prob, Y_hat, _, _ = model(features)

            # 记录准确率
            acc_logger.log(Y_hat, label)
            
            # 计算损失
            loss = loss_fn(logits, label)

            # 保存预测概率和标签
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, label)
            val_error += error
            
    # 计算平均损失和错误率
    val_error /= len(loader)
    val_loss /= len(loader)

    # 计算AUC
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
    
    # 记录验证指标到tensorboard
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)

    # 打印验证结果
    print('\n验证集结果:')
    print(f'验证损失: {val_loss:.4f}, 验证错误率: {val_error:.4f}, AUC: {auc:.4f}')
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))     

    # 执行早停检查
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return val_loss, val_error, auc

    return val_loss, val_error, auc

def validate_clam(cur, epoch, model, loader, n_classes, early_stopping = None, writer = None, loss_fn = None, results_dir = None):
    """CLAM模型的验证
    Args:
        cur: 当前fold索引
        epoch: 当前epoch
        model: 模型
        loader: 数据加载器
        n_classes: 类别数
        early_stopping: 早停对象
        writer: tensorboard写入器
        loss_fn: 损失函数
        results_dir: 结果保存目录
    Returns:
        bool: 是否触发早停
    """
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    inst_logger = Accuracy_Logger(n_classes=n_classes)
    val_loss = 0.
    val_error = 0.

    val_inst_loss = 0.
    val_inst_acc = 0.
    inst_count=0
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))
    sample_size = model.k_sample
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with torch.inference_mode():
        for batch_idx, data in enumerate(loader):
            if isinstance(data, tuple) and len(data) == 3:  # 如果是包含文本描述的数据
                features, label, text_desc = data
                features, label = features.to(device), label.to(device)
                
                # 前向传播（传入文本描述）
                logits, Y_prob, Y_hat, _, instance_dict = model(features, text_descriptions=text_desc, label=label, instance_eval=True)
            else:  # 如果是不包含文本描述的数据
                features, label = data
                features, label = features.to(device), label.to(device)
                
                # 前向传播（不传入文本描述）
                logits, Y_prob, Y_hat, _, instance_dict = model(features, label=label, instance_eval=True)
            
            # 记录准确率
            acc_logger.log(Y_hat, label)
            
            # 计算包级别损失
            loss = loss_fn(logits, label)
            val_loss += loss.item()

            # 计算实例级别损失
            instance_loss = instance_dict['instance_loss']
            
            inst_count+=1
            instance_loss_value = instance_loss.item()
            val_inst_loss += instance_loss_value

            # 记录实例级别预测
            inst_preds = instance_dict['inst_preds']
            inst_labels = instance_dict['inst_labels']
            inst_logger.log_batch(inst_preds, inst_labels)

            # 保存预测概率和标签
            prob[batch_idx] = Y_prob.cpu().numpy()
            labels[batch_idx] = label.item()
            
            error = calculate_error(Y_hat, label)
            val_error += error

    # 计算平均损失和错误率
    val_error /= len(loader)
    val_loss /= len(loader)

    # 计算AUC
    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], prob[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    # 打印验证结果
    print('\n验证集结果:')
    print(f'验证损失: {val_loss:.4f}, 验证错误率: {val_error:.4f}, AUC: {auc:.4f}')
    if inst_count > 0:
        val_inst_loss /= inst_count
        for i in range(2):
            acc, correct, count = inst_logger.get_summary(i)
            print('class {} clustering acc {}: correct {}/{}'.format(i, acc, correct, count))
    
    # 记录验证指标到tensorboard
    if writer:
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/auc', auc, epoch)
        writer.add_scalar('val/error', val_error, epoch)
        writer.add_scalar('val/inst_loss', val_inst_loss, epoch)

    # 打印每个类别的准确率
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        
        if writer and acc is not None:
            writer.add_scalar('val/class_{}_acc'.format(i), acc, epoch)
     
    # 执行早停检查
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return val_loss, val_error, auc

    return val_loss, val_error, auc

def summary(model, loader, n_classes):
    """模型评估总结
    Args:
        model: 模型
        loader: 数据加载器
        n_classes: 类别数
    Returns:
        patient_results: 每个病人的预测结果
        test_error: 测试错误率
        auc: ROC AUC值
        acc_logger: 准确率记录器
    """
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    slide_ids = loader.dataset.slide_data['slide_id']
    patient_results = {}

    for batch_idx, data in enumerate(loader):
        # 检查是否包含文本描述
        if len(data) == 3:  # 如果包含文本描述
            features, label, text_desc = data
            features, label = features.to(device), label.to(device)
            
            # 前向传播（传入文本描述）
            with torch.inference_mode():
                logits, Y_prob, Y_hat, _, _ = model(features, text_descriptions=text_desc)
        else:  # 如果不包含文本描述
            features, label = data
            features, label = features.to(device), label.to(device)
            
            # 前向传播（不传入文本描述）
            with torch.inference_mode():
                logits, Y_prob, Y_hat, _, _ = model(features)

        # 记录准确率
        acc_logger.log(Y_hat, label)
        
        # 保存预测结果
        probs = Y_prob.cpu().numpy()
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = label.item()
        
        slide_id = slide_ids.iloc[batch_idx]
        patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': label.item()}})
        error = calculate_error(Y_hat, label)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))

    return patient_results, test_error, auc, acc_logger


def get_split_loader(split_dataset, training = False, testing = False, weighted = False):
    """
    获取数据加载器
    Args:
        split_dataset: 数据集
        training: 是否为训练模式
        testing: 是否为测试模式
        weighted: 是否使用加权采样
    Returns:
        loader: 数据加载器
    """
    kwargs = {'num_workers': 4} if device.type == "cuda" else {}
    if not testing and weighted:
        weights = make_weights_for_balanced_classes_split(split_dataset)
        loader = DataLoader(split_dataset, batch_size=1, sampler = WeightedRandomSampler(weights, len(weights)), collate_fn=collate_features, **kwargs)
    else:
        loader = DataLoader(split_dataset, batch_size=1, sampler = None, collate_fn=collate_features, **kwargs)
    return loader

def collate_features(batch):
    """
    自定义的collate函数，处理包含文本描述的batch
    """
    if len(batch[0]) == 3:  # 如果包含文本描述
        features = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        text_descs = [item[2] for item in batch]
        return torch.cat(features, dim=0), torch.tensor(labels), text_descs
    else:  # 如果不包含文本描述
        features = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return torch.cat(features, dim=0), torch.tensor(labels)
