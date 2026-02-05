from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # ===== 必须正确的（你当前会用到） =====
    settings.got10k_path = r"/home/m1n1ons/projects/dev/DUTrack/data/got10k"
    settings.prj_dir = r"/home/m1n1ons/projects/dev/DUTrack"
    settings.save_dir = r"/home/m1n1ons/projects/dev/DUTrack/output"
    settings.results_path = r"/home/m1n1ons/projects/dev/DUTrack/output/test/tracking_results"
    settings.network_path = r"/home/m1n1ons/projects/dev/DUTrack/output/test/networks"
    settings.result_plot_path = r"/home/m1n1ons/projects/dev/DUTrack/output/test/result_plots"
    settings.segmentation_path = r"/home/m1n1ons/projects/dev/DUTrack/output/test/segmentation_results"

    # ===== 下面这些：现在不会用到，但建议也顺手改掉，避免以后踩坑 =====
    settings.got10k_lmdb_path = r"/home/m1n1ons/projects/dev/DUTrack/data/got10k_lmdb"
    settings.itb_path = r"/home/m1n1ons/projects/dev/DUTrack/data/itb"
    settings.lasot_extension_subset_path = r"/home/m1n1ons/projects/dev/DUTrack/data/lasot_extension_subset"
    settings.lasot_lmdb_path = r"/home/m1n1ons/projects/dev/DUTrack/data/lasot_lmdb"
    settings.lasot_path = r"/home/m1n1ons/projects/dev/DUTrack/data/lasot"
    settings.mgit_path = r"/home/m1n1ons/projects/dev/DUTrack/data/MGIT"
    settings.nfs_path = r"/home/m1n1ons/projects/dev/DUTrack/data/nfs"
    settings.otb_lang_path = r"/home/m1n1ons/projects/dev/DUTrack/data/OTB_sentences"
    settings.otb_path = r"/home/m1n1ons/projects/dev/DUTrack/data/otb"
    settings.tc128_path = r"/home/m1n1ons/projects/dev/DUTrack/data/TC128"
    settings.tnl2k_path = r"/home/m1n1ons/projects/dev/DUTrack/data/TNL2K"
    settings.trackingnet_path = r"/home/m1n1ons/projects/dev/DUTrack/data/trackingnet"
    settings.uav_path = r"/home/m1n1ons/projects/dev/DUTrack/data/uav"
    settings.vot18_path = r"/home/m1n1ons/projects/dev/DUTrack/data/vot2018"
    settings.vot22_path = r"/home/m1n1ons/projects/dev/DUTrack/data/vot2022"
    settings.vot_path = r"/home/m1n1ons/projects/dev/DUTrack/data/VOT2019"

    # ===== 可以暂时留空的 =====
    settings.davis_dir = ''
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.youtubevos_dir = ''

    return settings
