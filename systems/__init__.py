

import numpy as np
import logging

from . import btsewrapper
from . import aasistsslwrapper
from . import vocosigwrapper
from . import conformerwrapper

from . import config

# logging.basicConfig(filename='running.log', level=logging.DEBUG, format='[Decision] %(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)
# Create a new logging handler with your desired format
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)

SUPPORT_CM = ['aasist_ssl', 'vocosig', 'btse_wav2vec', 'conformer']

if config.BTSE_CONFIG["enable"]:
    btse_wav2vec = btsewrapper.BTSEWrapper(
        model_path=config.BTSE_CONFIG["model_path"],
        config_path=config.BTSE_CONFIG["config_path"],
        threshold=config.BTSE_CONFIG["threshold"]
    )
    logger.info("Finished loading btse model")
    btse_wav2vec_detect = btse_wav2vec.detect


if config.AASISTSSL_CONFIG["enable"]:
    aasist_ssl = aasistsslwrapper.AasistSSLWrapper(
        model_path=config.AASISTSSL_CONFIG["model_path"],
        threshold=config.AASISTSSL_CONFIG["threshold"]
    )
    logger.info("Finished loading aasistssl model")
    aasist_ssl_detect = aasist_ssl.detect

if config.VOCOSIG_CONFIG["enable"]:
    vocosig = vocosigwrapper.VocoSigWrapper(
        model_path=config.VOCOSIG_CONFIG["model_path"],
        config_path=config.VOCOSIG_CONFIG["config_path"],
        threshold=config.VOCOSIG_CONFIG["threshold"]
    )
    logger.info("Finished loading vocosig model")
    vocosig_detect = vocosig.detect

if config.CONFORMER_CONFIG["enable"]:
    conformer = conformerwrapper.ConformerWrapper(
        model_path=config.CONFORMER_CONFIG["model_path"],
        threshold=config.CONFORMER_CONFIG["threshold"]
    )
    logger.info("Finished loading conformer model")
    conformer_detect = conformer.detect