#!/bin/sh
gcloud container clusters get-credentials word-service --zone us-central1-c
gcloud endpoints services deploy ../../word_service/word_service_proto/api_descriptor.pb api_config.yaml
