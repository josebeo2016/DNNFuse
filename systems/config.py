BTSE_CONFIG = {
    'version': 'D-2312_M-BTSE-3',
    'model_path': '/dataa/phucdt/DNNFuse/systems/btse_model/models/model_weighted_CCE_100_8_1e-06_supcon_nov22_finetune_new_Trans_64concat/epoch_14.pth',
    'config_path': '/dataa/phucdt/DNNFuse/systems/btse_model/configs/model_config_RawNet_Trans_64concat.yaml',
    'threshold': -7.82,
    'enable': True,
}

AASISTSSL_CONFIG = {
    'version': 'D2312_M-AASIST-SSL',
    'model_path': '/dataa/phucdt/DNNFuse/systems/aasist_ssl/models/model_weighted_CCE_100_4_1e-06_supcon_nov22_new_finetune/epoch_47.pth',
    'threshold': 1.47,
    'enable': True,
}

CONFORMER_CONFIG = {
    'version': 'D2312_M-AASIST-SSL',
    'model_path': '/dataa/phucdt/DNNFuse/systems/aasist_ssl/models/model_weighted_CCE_100_10_1e-06_supcon_nov22_new_finetune_conformerer/epoch_23.pth',
    'threshold': 1.47,
    'enable': True,
}


VOCOSIG_CONFIG = {
    'version': 'D2312_M-VOCOSIG-5-R',
    'model_path': '/dataa/phucdt/DNNFuse/systems/scl_vocosig/pretrained/conf-5-linear-finetune-from_nov22-29-epoch49.pth',
    'config_path': '/dataa/phucdt/DNNFuse/systems/scl_vocosig/configs/conf-5-linear.yaml',
    'threshold': -4.4781,
    'enable': True,
}