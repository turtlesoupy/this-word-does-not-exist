export PYTHONPATH=.: 
export ASSET_PATH=/mnt/evo/projects/title-maker-pro 
python word_service/wordservice_server.py \
 --forward-model-path $ASSET_PATH/models/forward-dictionary-model-v1 \
 --inverse-model-path $ASSET_PATH/models/inverse-dictionary-model-v1 \
 --blacklist-path $ASSET_PATH/models/blacklist.pickle \
 --quantize \
 --device cpu
