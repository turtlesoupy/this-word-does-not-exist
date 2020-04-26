#!/bin/sh
: ${TWITTER_API_KEY?Need a value}
: ${TWITTER_API_SECRET?Need a value}

twurl authorize --consumer-key $TWITTER_API_KEY  --consumer-secret $TWITTER_API_SECRET