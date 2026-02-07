#!/usr/bin/env python
"""
DUTrack V4测试和对比脚本

用法:
    # 测试V4b版本
    python tracking/test_v4.py --tracker v4b --dataset got10k_val
    
    # 测试并对比
    python tracking/test_v4.py --compare --dataset got10k_val
    
    # 只分析已有结果
    python tracking/test_v4.py --analyze
"""

import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)


def get_tracker_config(version):
    """获取tracker和配置的映射"""
    configs = {
        'baseline': ('dutrack', 'dutrack_256_got_baseline'),
        'enhanced': ('dutrack_enhanced', 'dutrack_256_got_enhanced'),
        'v4b': ('dutrack_v4b', 'dutrack_256_got_v4b'),
        'v5c': ('dutrack_v4b', 'dutrack_256_got_v5c'),
        'v5e': ('dutrack_v5e', 'dutrack_256_got_v5e'),
    }
    return configs.get(version)


def run_test(tracker_name, config_name, dataset_name='got10k_val', threads=4, num_gpus=1):
    """运行跟踪测试"""
    from lib.test.evaluation import get_dataset
    from lib.test.evaluation.running import run_dataset
    from lib.test.evaluation.tracker import Tracker
    
    print(f"\n{'='*60}")
    print(f"测试: {tracker_name} / {config_name}")
    print(f"数据集: {dataset_name}")
    print(f"{'='*60}\n")
    
    dataset = get_dataset(dataset_name)
    trackers = [Tracker(tracker_name, config_name, dataset_name)]
    run_dataset(dataset, trackers, debug=0, threads=threads, num_gpus=num_gpus)


def analyze_results(versions, dataset_name='got10k_val'):
    """分析并比较结果"""
    from lib.test.evaluation import get_dataset
    from lib.test.analysis.plot_results import print_results
    from lib.test.evaluation.tracker import Tracker
    
    print(f"\n{'='*60}")
    print(f"分析结果对比")
    print(f"{'='*60}\n")
    
    trackers = []
    for version in versions:
        config = get_tracker_config(version)
        if config:
            tracker_name, config_name = config
            trackers.append(Tracker(tracker_name, config_name, dataset_name,
                                   display_name=f'{version.upper()}'))
    
    dataset = get_dataset(dataset_name)
    print_results(trackers, dataset, dataset_name, 
                  merge_results=True, 
                  plot_types=('success', 'prec'))


def main():
    parser = argparse.ArgumentParser(description='DUTrack V4测试和对比脚本')
    parser.add_argument('--tracker', type=str, choices=['baseline', 'enhanced', 'v4b', 'v5c', 'v5e'],
                        help='要测试的tracker版本')
    parser.add_argument('--dataset', type=str, default='got10k_val',
                        help='数据集名称 (default: got10k_val)')
    parser.add_argument('--compare', action='store_true',
                        help='测试并比较所有版本')
    parser.add_argument('--analyze', action='store_true',
                        help='只分析已有结果')
    parser.add_argument('--threads', type=int, default=1,
                        help='线程数 (default: 1)')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='GPU数量 (default: 1)')
    
    args = parser.parse_args()
    
    if args.analyze:
        # 只分析已有结果
        versions = ['baseline', 'v4b']
        analyze_results(versions, args.dataset)
    elif args.compare:
        # 测试并比较所有版本
        versions = ['baseline', 'v4b']
        for version in versions:
            config = get_tracker_config(version)
            if config:
                run_test(config[0], config[1], args.dataset, args.threads, args.num_gpus)
        analyze_results(versions, args.dataset)
    elif args.tracker:
        # 测试指定版本
        config = get_tracker_config(args.tracker)
        if config:
            run_test(config[0], config[1], args.dataset, args.threads, args.num_gpus)
        else:
            print(f"未知的tracker版本: {args.tracker}")
    else:
        print("请指定 --tracker, --compare, 或 --analyze")
        parser.print_help()


if __name__ == '__main__':
    main()
