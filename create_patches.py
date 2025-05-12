# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage # 导入全幻灯片图像处理类
from wsi_core.wsi_utils import StitchPatches # 导入图像块拼接工具
from wsi_core.batch_process_utils import initialize_df # 导入数据框初始化工具
# other imports
import os # 操作系统接口
import numpy as np # 数值计算库
import time # 时间处理
import argparse # 命令行参数解析
import pdb # Python调试器
import pandas as pd # 数据分析库

def stitching(file_path, downscale = 64):
    """
    拼接图像块生成热图
    Args:
        file_path: h5文件路径
        downscale: 下采样倍数
    Returns:
        heatmap: 生成的热图
        total_time: 处理耗时
    """
    start = time.time()
    heatmap = StitchPatches(file_path, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
    total_time = time.time() - start
    
    return heatmap, total_time

def segment(WSI_object, seg_params, filter_params):
    """
    对WSI进行组织分割
    Args:
        WSI_object: WSI对象
        seg_params: 分割参数
        filter_params: 过滤参数
    Returns:
        WSI_object: 处理后的WSI对象
        seg_time_elapsed: 分割耗时
    """
    ### Start Seg Timer
    start_time = time.time()

    # Segment
    WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time   
    return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
    """
    将WSI切分成小块并保存为h5格式
    Args:
        WSI_object: WSI对象
        **kwargs: 切块参数
    Returns:
        file_path: 保存的h5文件路径
        patch_time_elapsed: 切块耗时
    """
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.createPatches_bag_hdf5(**kwargs, save_coord=True)

    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed

def seg_and_patch(source, save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, 
                  patch_size = 256, step_size = 256, custom_downsample=1, 
                  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'},
                  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8 }, 
                  vis_params = {'vis_level': -1, 'line_thickness': 250},
                  patch_params = {'white_thresh': 5, 'black_thresh': 40, 'use_padding': True, 'contour_fn': 'four_pt'},
                  patch_level = 0,
                  use_default_params = False, 
                  seg = False, save_mask = True, 
                  stitch= False, 
                  patch = False, auto_skip=True, process_list = None):
    """
    主处理函数,执行分割和切块
    Args:
        source: 源文件目录
        save_dir: 保存目录
        patch_save_dir: 切块保存目录
        mask_save_dir: 掩码保存目录
        stitch_save_dir: 拼接图保存目录
        patch_size: 切块大小
        step_size: 步长
        custom_downsample: 自定义下采样倍数
        seg_params: 分割参数
        filter_params: 过滤参数
        vis_params: 可视化参数
        patch_params: 切块参数
        patch_level: 切块层级
        use_default_params: 是否使用默认参数
        seg: 是否执行分割
        save_mask: 是否保存掩码
        stitch: 是否执行拼接
        patch: 是否执行切块
        auto_skip: 是否自动跳过已处理文件
        process_list: 处理列表文件
    Returns:
        seg_times: 平均分割时间
        patch_times: 平均切块时间
    """

    # 获取并排序所有幻灯片文件
    slides = sorted(os.listdir(source))
    slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]

    # 初始化或读取处理数据框
    if process_list is None:
        df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params, save_patches=True)
    
    else:
        df = pd.read_csv(process_list)
        df = initialize_df(df, seg_params, filter_params, vis_params, patch_params, save_patches=True)

    # 获取待处理的幻灯片
    mask = df['process'] == 1
    process_stack = df[mask]

    total = len(process_stack)
    seg_times = 0.
    patch_times = 0.
    stitch_times = 0.

    # 主处理循环
    for i in range(total):
        df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
        idx = process_stack.index[i]
        slide = process_stack.loc[idx, 'slide_id']
        print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
        print('processing {}'.format(slide))
        
        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide)

        # 检查是否需要跳过已处理文件
        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exist in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # 初始化WSI对象
        full_path = os.path.join(source, slide)
        WSI_object = WholeSlideImage(full_path)

        # 设置处理参数
        if use_default_params:
            current_vis_params = vis_params.copy()
            current_filter_params = filter_params.copy()
            current_seg_params = seg_params.copy()
            current_patch_params = patch_params.copy()
            
        else:
            current_vis_params = {}
            current_filter_params = {}
            current_seg_params = {}
            current_patch_params = {}
            for key in vis_params.keys():
                current_vis_params.update({key: df.loc[idx, key]})

            for key in filter_params.keys():
                current_filter_params.update({key: df.loc[idx, key]})

            for key in seg_params.keys():
                current_seg_params.update({key: df.loc[idx, key]})

            for key in patch_params.keys():
                current_patch_params.update({key: df.loc[idx, key]})

        # 设置可视化层级
        if current_vis_params['vis_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_vis_params['vis_level'] = 0
            
            else:    
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_vis_params['vis_level'] = best_level

        # 设置分割层级
        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0
            
            else:
                wsi = WSI_object.getOpenSlide()
                best_level = wsi.get_best_level_for_downsample(64)
                current_seg_params['seg_level'] = best_level

        # 处理保留的区域ID
        keep_ids = str(current_seg_params['keep_ids'])
        if keep_ids != 'none' and len(keep_ids) > 0:
            str_ids = current_seg_params['keep_ids']
            current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['keep_ids'] = []

        # 处理排除的区域ID
        exclude_ids = str(current_seg_params['exclude_ids'])
        if exclude_ids != 'none' and len(exclude_ids) > 0:
            str_ids = current_seg_params['exclude_ids']
            current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
        else:
            current_seg_params['exclude_ids'] = []

        # 检查图像尺寸是否过大
        w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
        if w * h > 1e8:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        # 更新数据框中的层级信息
        if not process_list:
            df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
            df.loc[idx, 'seg_level'] = current_seg_params['seg_level']

        # 执行分割
        seg_time_elapsed = -1
        if seg:
            WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

        # 保存掩码
        if save_mask:
            mask = WSI_object.visWSI(**current_vis_params)
            mask_path = os.path.join(mask_save_dir, slide_id+'.png')
            mask.save(mask_path)

        # 执行切块
        patch_time_elapsed = -1 # Default time
        if patch:
            current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 
                                         'save_path': patch_save_dir, 'custom_downsample': custom_downsample})
            file_path, patch_time_elapsed = patching(WSI_object = WSI_object, **current_patch_params)
        
        # 执行拼接
        stitch_time_elapsed = -1
        if stitch:
            file_path = os.path.join(patch_save_dir, slide_id+'.h5')
            heatmap, stitch_time_elapsed = stitching(file_path, downscale=64)
            stitch_path = os.path.join(stitch_save_dir, slide_id+'.png')
            heatmap.save(stitch_path)

        # 输出处理时间
        print("segmentation took {} seconds".format(seg_time_elapsed))
        print("patching took {} seconds".format(patch_time_elapsed))
        print("stitching took {} seconds".format(stitch_time_elapsed))
        df.loc[idx, 'status'] = 'processed'

        # 累计处理时间
        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed
        stitch_times += stitch_time_elapsed

    # 计算平均处理时间
    seg_times /= total
    patch_times /= total
    stitch_times /= total

    # 保存处理结果
    df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
    print("average stiching time in s per slide: {}".format(stitch_times))
        
    return seg_times, patch_times

# 设置命令行参数解析器
parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str,
                    help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256,
                    help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
                    help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
                    help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
                    help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, 
                    help='downsample level at which to patch')
parser.add_argument('--custom_downsample', type= int, choices=[1,2], default=1, 
                    help='custom downscale when native downsample is not available (only tested w/ 2x downscale)')
parser.add_argument('--process_list',  type = str, default=None,
                    help='name of list of images to process with parameters (.csv)')

if __name__ == '__main__':
    # 解析命令行参数
    args = parser.parse_args()

    # 设置保存目录
    patch_save_dir = os.path.join(args.save_dir, 'patches')
    mask_save_dir = os.path.join(args.save_dir, 'masks')
    stitch_save_dir = os.path.join(args.save_dir, 'stitches')

    # 设置处理列表
    if args.process_list:
        process_list = os.path.join(args.save_dir, args.process_list)
    else:
        process_list = None

    # 打印目录信息
    print('source: ', args.source)
    print('patch_save_dir: ', patch_save_dir)
    print('mask_save_dir: ', mask_save_dir)
    print('stitch_save_dir: ', stitch_save_dir)
    
    # 创建目录字典
    directories = {'source': args.source, 
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir, 
                   'mask_save_dir' : mask_save_dir, 
                   'stitch_save_dir': stitch_save_dir} 

    # 创建必要的目录
    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    # 设置默认参数
    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
                  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8 }
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'white_thresh': 5, 'black_thresh': 40, 'use_padding': True, 'contour_fn': 'four_pt'}

    # 如果有预设文件，从预设文件加载参数
    if args.preset:
        preset_df = pd.read_csv(os.path.join('presets', args.preset))
        for key in seg_params.keys():
            seg_params[key] = preset_df.loc[0, key]

        for key in filter_params.keys():
            filter_params[key] = preset_df.loc[0, key]

        for key in vis_params.keys():
            vis_params[key] = preset_df.loc[0, key]

        for key in patch_params.keys():
            patch_params[key] = preset_df.loc[0, key]
    
    # 创建参数字典
    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                  'vis_params': vis_params}

    print(parameters)

    # 执行主处理函数
    seg_times, patch_times = seg_and_patch(**directories, **parameters,
                                            patch_size = args.patch_size, step_size=args.step_size, 
                                            seg = args.seg,  use_default_params=False, save_mask = True, 
                                            stitch= args.stitch, custom_downsample = args.custom_downsample, 
                                            patch_level=args.patch_level, patch = args.patch,
                                            process_list = process_list, auto_skip=args.no_auto_skip)
