#!/bin/bash
export ASSET_PATH = /home/tdimson/title-maker-pro/build
export GOOGLE_APPLICATION_CREDENTIALS=$ASSET_PATH/this-word-does-not-exist-a55cb3814f2b.json

. /home/tdimson/title-maker-pro/build/env_vars.sh && \
/home/tdimson/miniconda3/envs/title_maker_pro/bin/python \
/home/tdimson/title-maker-pro/title_maker_pro/wotd_bot.py \
--log-file /home/tdimson/title-maker-pro/logs/wotd_gcloud.log \
--gcloud-project this-word-does-not-exist \
--gcloud-bucket this-word-does-not-exist-wotd \
--forward-model-path $ASSET_PATH/forward-dictionary-model-v1 \
--blacklist-path $ASSET_PATH/blacklist.pickle \
--inverse-model-path $ASSET_PATH/inverse-dictionary-model-v1 \
$@