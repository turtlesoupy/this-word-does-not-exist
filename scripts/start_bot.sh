. /home/tdimson/title-maker-pro/build/env_vars.sh && \
/home/tdimson/miniconda3/envs/title_maker_pro/bin/python \
/home/tdimson/title-maker-pro/title_maker_pro/twitter_bot.py \
--forward-model-path /home/tdimson/title-maker-pro/build/forward-dictionary-model-v1 \
--blacklist-path /home/tdimson/title-maker-pro/build/blacklist.pickle \
--log-file /home/tdimson/title-maker-pro/logs/bot.log \
--inverse-model-path /home/tdimson/title-maker-pro/build/inverse-dictionary-model-v1 
